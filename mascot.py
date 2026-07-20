"""Tsunao mascot state machine and interactive desktop widget."""

from __future__ import annotations

import ctypes
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSequentialAnimationGroup,
    Qt,
    QTimer,
    Property,
    Signal,
)
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QRegion
from PySide6.QtWidgets import QApplication, QWidget


class TsunaoState(str, Enum):
    SLEEPING = "sleeping"
    AWAKE = "awake"
    STANDING = "standing"
    WALKING = "walking"
    SMUG = "smug"
    TOUCHEAR = "touchear"
    DROWSY = "drowsy"
    DRAWING = "drawing"
    COMPLETE = "complete"
    PICKED = "picked"
    ANGRY = "angry"
    FLYING = "flying"
    HURT = "hurt"


class TsunaoJobState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETE = "complete"


DEFAULT_STATES = {
    TsunaoState.STANDING,
    TsunaoState.WALKING,
    TsunaoState.SMUG,
    TsunaoState.TOUCHEAR,
    TsunaoState.DROWSY,
}
INTERACTION_STATES = {
    TsunaoState.PICKED,
    TsunaoState.ANGRY,
    TsunaoState.FLYING,
    TsunaoState.HURT,
}


@dataclass(frozen=True)
class _RoutineSnapshot:
    state: TsunaoState
    elapsed_ms: float
    default_round: int
    walking_leg: int


class TsunaoStateMachine:
    """Pure pose FSM; movement and rendering live in :class:`TsunaoWidget`."""

    AWAKE_DURATION_MS = 5_000
    COMPLETE_DURATION_MS = 5_000
    ANGRY_DURATION_MS = 3_000
    HURT_DURATION_MS = 5_000
    STANDING_DURATION_MS = 3_000
    WALKING_MIN_DURATION_MS = 5_000
    WALKING_MAX_DURATION_MS = 10_000
    STATIC_DEFAULT_DURATION_MS = 5_000
    TOUCHEAR_DURATION_MS = 3_000
    DROWSY_DURATION_MS = 10_000
    DEFAULT_ROUNDS = 2

    def __init__(self, *, random_source: random.Random | None = None) -> None:
        self._random = random_source or random.Random()
        self.state = TsunaoState.SLEEPING
        self.job_state = TsunaoJobState.IDLE
        self.elapsed_ms = 0.0
        self._walking_duration_ms = float(self.WALKING_MIN_DURATION_MS)
        self.default_round = 0
        self.walking_leg = 1
        self.flying_frame = 0
        self._resume_snapshot: _RoutineSnapshot | None = None
        self._job_resume_snapshot: _RoutineSnapshot | None = None

    def image_loaded(self) -> bool:
        """Wake only when the currently visible pose is sleeping."""
        if self.state != TsunaoState.SLEEPING:
            return False
        self._set_state(TsunaoState.AWAKE)
        return True

    def processing_started(self) -> bool:
        self.job_state = TsunaoJobState.PROCESSING
        if self.state in INTERACTION_STATES:
            return False
        self._job_resume_snapshot = self._snapshot()
        self._set_state(TsunaoState.DRAWING)
        return True

    def processing_finished(self, *, completed: bool) -> bool:
        self.job_state = (
            TsunaoJobState.COMPLETE if completed else TsunaoJobState.IDLE
        )
        if self.state in INTERACTION_STATES:
            return False
        if completed:
            self._job_resume_snapshot = None
            self._set_state(TsunaoState.COMPLETE)
        else:
            self._restore_after_cancelled_job()
        return True

    def pick_up(self) -> bool:
        if self.state == TsunaoState.PICKED:
            return False
        if self.state not in INTERACTION_STATES:
            self._resume_snapshot = self._snapshot()
        elif self._resume_snapshot is None:
            self._resume_snapshot = _RoutineSnapshot(
                TsunaoState.STANDING,
                0.0,
                0,
                1,
            )
        self._set_state(TsunaoState.PICKED)
        return True

    def release(self, *, fast: bool) -> bool:
        if self.state != TsunaoState.PICKED:
            return False
        if fast:
            self.flying_frame = 0
            self._set_state(TsunaoState.FLYING)
        else:
            self._set_state(TsunaoState.ANGRY)
        return True

    def wall_hit(self) -> bool:
        if self.state != TsunaoState.FLYING:
            return False
        self.flying_frame = (self.flying_frame + 1) % 4
        return True

    def flight_stopped(self) -> bool:
        if self.state != TsunaoState.FLYING:
            return False
        self._set_state(TsunaoState.HURT)
        return True

    def advance(self, milliseconds: float) -> bool:
        """Advance timers deterministically, including across phase boundaries."""
        remaining = max(0.0, float(milliseconds))
        changed = False
        while remaining > 0:
            duration = self._state_duration_ms()
            if duration is None:
                self.elapsed_ms += remaining
                break

            until_transition = max(0.0, duration - self.elapsed_ms)
            step = min(remaining, until_transition)
            self.elapsed_ms += step
            remaining -= step
            if self.elapsed_ms + 1e-6 < duration:
                break
            self._handle_timeout()
            changed = True
        return changed

    def _state_duration_ms(self) -> float | None:
        durations = {
            TsunaoState.AWAKE: self.AWAKE_DURATION_MS,
            TsunaoState.COMPLETE: self.COMPLETE_DURATION_MS,
            TsunaoState.ANGRY: self.ANGRY_DURATION_MS,
            TsunaoState.HURT: self.HURT_DURATION_MS,
            TsunaoState.STANDING: self.STANDING_DURATION_MS,
            TsunaoState.WALKING: self._walking_duration_ms,
            TsunaoState.SMUG: self.STATIC_DEFAULT_DURATION_MS,
            TsunaoState.TOUCHEAR: self.TOUCHEAR_DURATION_MS,
            TsunaoState.DROWSY: self.DROWSY_DURATION_MS,
        }
        return durations.get(self.state)

    def _handle_timeout(self) -> None:
        if self.state == TsunaoState.AWAKE:
            self._start_default_routine()
        elif self.state == TsunaoState.COMPLETE:
            self.job_state = TsunaoJobState.IDLE
            self._start_default_routine()
        elif self.state in {TsunaoState.ANGRY, TsunaoState.HURT}:
            self._resolve_interruption()
        elif self.state == TsunaoState.STANDING:
            self.walking_leg = 1
            self._set_state(TsunaoState.WALKING)
        elif self.state == TsunaoState.WALKING:
            self._set_state(
                TsunaoState.SMUG
                if self.walking_leg == 1
                else TsunaoState.TOUCHEAR
            )
        elif self.state == TsunaoState.SMUG:
            self.walking_leg = 2
            self._set_state(TsunaoState.WALKING)
        elif self.state == TsunaoState.TOUCHEAR:
            if self.default_round + 1 >= self.DEFAULT_ROUNDS:
                self._set_state(TsunaoState.DROWSY)
            else:
                self.default_round += 1
                self.walking_leg = 1
                self._set_state(TsunaoState.WALKING)
        elif self.state == TsunaoState.DROWSY:
            self._set_state(TsunaoState.SLEEPING)

    def _start_default_routine(self) -> None:
        self.default_round = 0
        self.walking_leg = 1
        self._resume_snapshot = None
        self._job_resume_snapshot = None
        self._set_state(TsunaoState.STANDING)

    def _resolve_interruption(self) -> None:
        if self.job_state == TsunaoJobState.PROCESSING:
            self._set_state(TsunaoState.DRAWING)
        elif self.job_state == TsunaoJobState.COMPLETE:
            self._set_state(TsunaoState.COMPLETE)
        else:
            snapshot = self._job_resume_snapshot or self._resume_snapshot
            if snapshot is None or snapshot.state in {
                TsunaoState.DRAWING,
                TsunaoState.SLEEPING,
            }:
                self._start_default_routine()
                return
            self._restore(snapshot, restart_transient_timer=True)
        self._resume_snapshot = None
        self._job_resume_snapshot = None

    def _restore_after_cancelled_job(self) -> None:
        snapshot = self._job_resume_snapshot
        self._job_resume_snapshot = None
        if snapshot is None or snapshot.state == TsunaoState.DRAWING:
            self._start_default_routine()
            return
        self._restore(snapshot, restart_transient_timer=False)

    def _snapshot(self) -> _RoutineSnapshot:
        return _RoutineSnapshot(
            self.state,
            self.elapsed_ms,
            self.default_round,
            self.walking_leg,
        )

    def _restore(
        self,
        snapshot: _RoutineSnapshot,
        *,
        restart_transient_timer: bool,
    ) -> None:
        self.state = snapshot.state
        self.elapsed_ms = (
            0.0
            if restart_transient_timer
            and snapshot.state in {TsunaoState.AWAKE, TsunaoState.COMPLETE}
            else snapshot.elapsed_ms
        )
        self.default_round = snapshot.default_round
        self.walking_leg = snapshot.walking_leg

    def _set_state(self, state: TsunaoState) -> None:
        self.state = state
        self.elapsed_ms = 0.0
        if state == TsunaoState.WALKING:
            self._walking_duration_ms = self._random.uniform(
                self.WALKING_MIN_DURATION_MS,
                self.WALKING_MAX_DURATION_MS,
            )


MASCOT_ASSET_FILES: dict[TsunaoState, tuple[str, ...]] = {
    TsunaoState.SLEEPING: ("asleep.png", "asleep-2.png"),
    TsunaoState.AWAKE: ("awake-1.png", "awake-2.png"),
    TsunaoState.STANDING: ("standing-1.png", "standing-2.png"),
    TsunaoState.WALKING: ("walking-1.png", "walking-2.png"),
    TsunaoState.SMUG: ("smug-1.png", "smug-2.png"),
    TsunaoState.TOUCHEAR: ("touchear-1.png", "touchear-2.png"),
    TsunaoState.DROWSY: ("drowsy-1.png", "drowsy-2.png"),
    TsunaoState.DRAWING: ("drawing-1.png", "drawing-2.png"),
    TsunaoState.COMPLETE: ("complete.png", "complete-2.png"),
    TsunaoState.PICKED: ("picked-1.png", "picked-2.png"),
    TsunaoState.ANGRY: ("angry-1.png", "angry-2.png"),
    TsunaoState.FLYING: (
        "flying-1.png",
        "flying-2.png",
        "flying-3.png",
        "flying-4.png",
    ),
    TsunaoState.HURT: ("hurt-1.png", "hurt-2.png"),
}

MASCOT_ACCESSIBLE_NAMES = {
    TsunaoState.SLEEPING: "잠든 츠나오",
    TsunaoState.AWAKE: "잠에서 깬 츠나오",
    TsunaoState.STANDING: "서 있는 츠나오",
    TsunaoState.WALKING: "천천히 걷는 츠나오",
    TsunaoState.SMUG: "뿌듯해하는 츠나오",
    TsunaoState.TOUCHEAR: "귀를 만지는 츠나오",
    TsunaoState.DROWSY: "졸고 있는 츠나오",
    TsunaoState.DRAWING: "그림을 그리며 처리 중인 츠나오",
    TsunaoState.COMPLETE: "완성된 그림을 보여주는 츠나오",
    TsunaoState.PICKED: "마우스에 들린 츠나오",
    TsunaoState.ANGRY: "화가 난 츠나오",
    TsunaoState.FLYING: "창 안을 날아가는 츠나오",
    TsunaoState.HURT: "벽에 부딪혀 아파하는 츠나오",
}

RANDOM_FLIP_STATES = {
    TsunaoState.SLEEPING,
    TsunaoState.AWAKE,
    TsunaoState.STANDING,
    TsunaoState.SMUG,
    TsunaoState.TOUCHEAR,
    TsunaoState.DROWSY,
    TsunaoState.DRAWING,
    TsunaoState.COMPLETE,
    TsunaoState.PICKED,
    TsunaoState.ANGRY,
    TsunaoState.HURT,
}


def _client_animations_enabled() -> bool:
    if os.name != "nt":
        return True
    try:
        enabled = ctypes.c_int()
        success = ctypes.windll.user32.SystemParametersInfoW(
            0x1042,  # SPI_GETCLIENTAREAANIMATION
            0,
            ctypes.byref(enabled),
            0,
        )
        return bool(enabled.value) if success else True
    except (AttributeError, OSError):
        return True


class HitEffectWidget(QWidget):
    """Wall impact effect rotating around the asset's bottom-center anchor."""

    def __init__(
        self,
        parent: QWidget,
        *,
        effect_size: int,
        animations_enabled: bool,
    ) -> None:
        super().__init__(parent)
        self._effect_size = effect_size
        self._pixmap = QPixmap()
        self._rotation = 0.0
        self._opacity = 0.0
        canvas_size = math.ceil(effect_size * 2.5)
        if canvas_size % 2:
            canvas_size += 1
        self.setFixedSize(canvas_size, canvas_size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()

        self._sequence = QSequentialAnimationGroup(self)
        fade_in = QPropertyAnimation(self, b"opacity", self._sequence)
        fade_in.setDuration(80)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        fade_out = QPropertyAnimation(self, b"opacity", self._sequence)
        fade_out.setDuration(210)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self._sequence.addAnimation(fade_in)
        self._sequence.addAnimation(fade_out)
        self._sequence.finished.connect(self.hide)

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(190)
        self._hold_timer.timeout.connect(self.hide)
        self._animations_enabled = animations_enabled

    def _get_opacity(self) -> float:
        return self._opacity

    def _set_opacity(self, opacity: float) -> None:
        self._opacity = self._clamp_opacity(opacity)
        self.update()

    opacity = Property(float, _get_opacity, _set_opacity)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap

    def show_effect(self, contact: QPointF, rotation: float) -> None:
        if self._pixmap.isNull():
            return
        self._sequence.stop()
        self._hold_timer.stop()
        self._rotation = rotation
        self.move(
            round(contact.x() - self.width() / 2),
            round(contact.y() - self.height() / 2),
        )
        self.show()
        self.raise_()
        if self._animations_enabled:
            self.opacity = 0.0
            self._sequence.start()
        else:
            self.opacity = 1.0
            self._hold_timer.start()

    def shutdown(self) -> None:
        self._sequence.stop()
        self._hold_timer.stop()
        self.hide()

    def paintEvent(self, event) -> None:
        del event
        if self._pixmap.isNull() or self._opacity <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(self._opacity)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._rotation)
        target = QRectF(
            -self._effect_size / 2,
            -self._effect_size,
            self._effect_size,
            self._effect_size,
        )
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

    @staticmethod
    def _clamp_opacity(opacity: float) -> float:
        return max(0.0, min(1.0, opacity))


class TsunaoWidget(QWidget):
    """Animated mascot that can be picked up and thrown around its parent."""

    state_changed = Signal(str)
    wall_hit = Signal(QPointF)

    DISPLAY_SIZE = 188
    HIT_SIZE = 118
    TICK_MS = 16
    WALK_SPEED = 96.0
    WALK_ACCELERATION = 520.0
    THROW_SPEED_THRESHOLD = 850.0
    THROW_GAIN = 2.10
    MAX_THROW_SPEED = 4_200.0
    LINEAR_DRAG = 0.90
    BOUNCE_RESTITUTION = 0.80
    STOP_SPEED = 42.0
    MAX_ANGULAR_SPEED = 560.0
    ANGULAR_DRAG = 1.20
    ANGLE_RESET_RATE = 24.0
    FOLLOW_RATE = 60.0

    _FRAME_INTERVALS_MS = {
        TsunaoState.SLEEPING: 950,
        TsunaoState.PICKED: 170,
        TsunaoState.ANGRY: 220,
        TsunaoState.WALKING: 240,
        TsunaoState.DRAWING: 290,
        TsunaoState.HURT: 340,
    }

    def __init__(
        self,
        parent: QWidget,
        *,
        animations_enabled: bool | None = None,
        random_source: random.Random | None = None,
    ) -> None:
        super().__init__(parent)
        self._animations_enabled = (
            _client_animations_enabled()
            if animations_enabled is None
            else animations_enabled
        )
        self._random = random_source or random.Random()
        self._machine = TsunaoStateMachine(random_source=self._random)
        self._frames: dict[TsunaoState, tuple[QPixmap, ...]] = {}
        self._frame_images: dict[int, QImage] = {}
        self._frame_bounds: dict[int, QRectF] = {}
        self._frame_index = 0
        self._frame_elapsed_ms = 0.0
        self._position: QPointF | None = None
        self._velocity = QPointF()
        self._angle = 0.0
        self._angular_velocity = 0.0
        self._angle_resetting = False
        self._walk_direction = 1
        self._walk_velocity = 0.0
        self._mirrored = False
        self._user_hidden = False
        self._drag_target: QPointF | None = None
        self._drag_samples: deque[tuple[float, QPointF]] = deque(maxlen=12)
        self._last_tick = time.monotonic()
        self._hit_pixmap = QPixmap()

        canvas_size = math.ceil(self.DISPLAY_SIZE * math.sqrt(2)) + 8
        self.setFixedSize(canvas_size, canvas_size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAccessibleName(MASCOT_ACCESSIBLE_NAMES[self.state])
        self.setAccessibleDescription(
            "마우스로 잡아 움직이거나 빠르게 던질 수 있는 장식 캐릭터"
        )
        self.hide()

        self._hit_effect = HitEffectWidget(
            parent,
            effect_size=self.HIT_SIZE,
            animations_enabled=self._animations_enabled,
        )

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    @property
    def state(self) -> TsunaoState:
        return self._machine.state

    @property
    def position(self) -> QPointF:
        self._ensure_initial_position()
        return QPointF(self._position)

    @property
    def is_dragging(self) -> bool:
        return self.state == TsunaoState.PICKED

    @property
    def mirrored(self) -> bool:
        return self._mirrored

    def set_assets(self, directory: str | Path) -> None:
        root = Path(directory)
        frames = {
            state: tuple(QPixmap(str(root / filename)) for filename in filenames)
            for state, filenames in MASCOT_ASSET_FILES.items()
        }
        hit = QPixmap(str(root / "hit.png"))
        if any(
            pixmap.isNull()
            for state_frames in frames.values()
            for pixmap in state_frames
        ) or hit.isNull():
            self._frames.clear()
            self._timer.stop()
            self._hit_effect.hide()
            self.hide()
            return

        self._frames = frames
        self._frame_images.clear()
        self._frame_bounds.clear()
        for state_frames in frames.values():
            for pixmap in state_frames:
                cache_key = pixmap.cacheKey()
                self._frame_images[cache_key] = pixmap.toImage().scaled(
                    self.DISPLAY_SIZE,
                    self.DISPLAY_SIZE,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                mask = pixmap.mask()
                mask_bounds = (
                    pixmap.rect()
                    if mask.isNull()
                    else QRegion(mask).boundingRect()
                )
                scale_x = self.DISPLAY_SIZE / max(1, pixmap.width())
                scale_y = self.DISPLAY_SIZE / max(1, pixmap.height())
                self._frame_bounds[cache_key] = QRectF(
                    mask_bounds.left() * scale_x - self.DISPLAY_SIZE / 2,
                    mask_bounds.top() * scale_y - self.DISPLAY_SIZE / 2,
                    mask_bounds.width() * scale_x,
                    mask_bounds.height() * scale_y,
                )
        self._hit_pixmap = hit
        self._hit_effect.set_pixmap(hit)
        self._set_orientation_for_state()
        self._ensure_initial_position()
        self._sync_geometry()
        if not self._user_hidden:
            self.show()
        self.raise_()
        self._timer.start()
        self.update()

    def set_user_hidden(self, hidden: bool) -> None:
        """사용자 설정에 따라 마스코트와 충돌 효과를 함께 숨긴다."""
        self._user_hidden = bool(hidden)
        if self._user_hidden:
            self._hit_effect.shutdown()
            self.hide()
            return
        if self._frames:
            self._sync_geometry()
            self.show()
            self.raise_()
            self.update()

    def image_loaded(self) -> None:
        previous = self.state
        self._machine.image_loaded()
        self._after_state_change(previous)

    def processing_started(self) -> None:
        previous = self.state
        self._machine.processing_started()
        self._after_state_change(previous)

    def processing_finished(self, *, completed: bool) -> None:
        previous = self.state
        self._machine.processing_finished(completed=completed)
        self._after_state_change(previous)

    def advance_time(self, milliseconds: float) -> None:
        """Advance pose timers without waiting; useful for deterministic checks."""
        previous = self.state
        self._machine.advance(milliseconds)
        self._after_state_change(previous)

    def hit_test(self, parent_position: QPointF) -> bool:
        if not self.isVisible() or not self._frames or self._position is None:
            return False
        pixmap = self._current_pixmap()
        if pixmap.isNull():
            return False

        local_x = parent_position.x() - self._position.x()
        local_y = parent_position.y() - self._position.y()
        radians = math.radians(-self._angle)
        unrotated_x = local_x * math.cos(radians) - local_y * math.sin(radians)
        unrotated_y = local_x * math.sin(radians) + local_y * math.cos(radians)
        if self._mirrored:
            unrotated_x = -unrotated_x
        source_x = int(unrotated_x + self.DISPLAY_SIZE / 2)
        source_y = int(unrotated_y + self.DISPLAY_SIZE / 2)
        if not (
            0 <= source_x < self.DISPLAY_SIZE
            and 0 <= source_y < self.DISPLAY_SIZE
        ):
            return False
        image = self._frame_images.get(pixmap.cacheKey())
        return image is not None and image.pixelColor(source_x, source_y).alpha() > 24

    def handle_mouse_event(self, event: QMouseEvent, parent_position: QPointF) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            if not self.hit_test(parent_position):
                return False
            self.begin_pick(parent_position)
            return True

        if event.type() == QEvent.Type.MouseMove and self.is_dragging:
            self.update_pick(parent_position)
            return True

        if event.type() == QEvent.Type.MouseButtonRelease and self.is_dragging:
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            self.release_pick(parent_position)
            return True
        return False

    def begin_pick(self, parent_position: QPointF) -> None:
        previous = self.state
        self._machine.pick_up()
        self._velocity = QPointF()
        self._angular_velocity = 0.0
        self._angle = 0.0
        self._angle_resetting = False
        self._drag_target = self._cursor_target(parent_position)
        now = time.monotonic()
        self._drag_samples.clear()
        self._drag_samples.append((now, QPointF(parent_position)))
        QApplication.setOverrideCursor(Qt.CursorShape.ClosedHandCursor)
        self._after_state_change(previous)

    def update_pick(self, parent_position: QPointF) -> None:
        self._drag_target = self._cursor_target(parent_position)
        now = time.monotonic()
        self._drag_samples.append((now, QPointF(parent_position)))
        while self._drag_samples and now - self._drag_samples[0][0] > 0.16:
            self._drag_samples.popleft()

    def release_pick(self, parent_position: QPointF) -> None:
        self.update_pick(parent_position)
        throw_velocity = self._sampled_pointer_velocity()
        speed = math.hypot(throw_velocity.x(), throw_velocity.y())
        fast = speed >= self.THROW_SPEED_THRESHOLD
        previous = self.state
        self._machine.release(fast=fast)
        self._drag_target = None
        QApplication.restoreOverrideCursor()

        if fast:
            boosted_velocity = QPointF(
                throw_velocity.x() * self.THROW_GAIN,
                throw_velocity.y() * self.THROW_GAIN,
            )
            boosted_speed = math.hypot(
                boosted_velocity.x(),
                boosted_velocity.y(),
            )
            scale = min(
                1.0,
                self.MAX_THROW_SPEED / max(boosted_speed, 1.0),
            )
            self._velocity = QPointF(
                boosted_velocity.x() * scale,
                boosted_velocity.y() * scale,
            )
            spin = (self._velocity.x() - self._velocity.y()) * 0.16
            if abs(spin) < 120.0:
                spin = 120.0 if self._velocity.x() >= 0 else -120.0
            self._angular_velocity = self._clamp(
                spin,
                -self.MAX_ANGULAR_SPEED,
                self.MAX_ANGULAR_SPEED,
            )
        else:
            self._velocity = QPointF()
            self._angular_velocity = 0.0
        self._after_state_change(previous)

    def parent_resized(self) -> None:
        self._ensure_initial_position()
        if self.state != TsunaoState.PICKED:
            self._keep_visible_inside_parent()
        self._sync_geometry()

    def place_initial(self, position: QPointF) -> None:
        self._position = QPointF(position)
        self._keep_visible_inside_parent()
        self._sync_geometry()

    def shutdown(self) -> None:
        self._timer.stop()
        self._hit_effect.shutdown()
        if self.is_dragging:
            QApplication.restoreOverrideCursor()
        self._drag_target = None

    def paintEvent(self, event) -> None:
        del event
        pixmap = self._current_pixmap()
        if pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)
        if self._mirrored:
            painter.scale(-1, 1)
        target = QRectF(
            -self.DISPLAY_SIZE / 2,
            -self.DISPLAY_SIZE / 2,
            self.DISPLAY_SIZE,
            self.DISPLAY_SIZE,
        )
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))

    def _tick(self) -> None:
        now = time.monotonic()
        delta_seconds = min(0.05, max(0.001, now - self._last_tick))
        self._last_tick = now
        previous = self.state
        self._machine.advance(delta_seconds * 1_000)
        self._after_state_change(previous)

        if self.state == TsunaoState.PICKED:
            self._advance_drag(delta_seconds)
        elif self.state == TsunaoState.WALKING and self._animations_enabled:
            self._advance_walk(delta_seconds)
        elif self.state == TsunaoState.FLYING:
            self._advance_flight(delta_seconds)

        self._advance_frame(delta_seconds * 1_000)
        self._sync_geometry()
        self.update()

    def _advance_drag(self, delta_seconds: float) -> None:
        if self._drag_target is None or self._position is None:
            return
        response = 1.0 - math.exp(-self.FOLLOW_RATE * delta_seconds)
        self._position = QPointF(
            self._position.x()
            + (self._drag_target.x() - self._position.x()) * response,
            self._position.y()
            + (self._drag_target.y() - self._position.y()) * response,
        )

    def _advance_walk(self, delta_seconds: float) -> None:
        if self._position is None:
            return
        target_velocity = self._walk_direction * self.WALK_SPEED
        velocity_change = self._clamp(
            target_velocity - self._walk_velocity,
            -self.WALK_ACCELERATION * delta_seconds,
            self.WALK_ACCELERATION * delta_seconds,
        )
        self._walk_velocity += velocity_change
        self._position.setX(
            self._position.x() + self._walk_velocity * delta_seconds
        )
        bounds = self._visible_local_bounds()
        parent_width = self.parentWidget().width()
        if self._position.x() + bounds.left() <= 0:
            self._position.setX(-bounds.left())
            self._walk_direction = 1
            self._walk_velocity = abs(self._walk_velocity)
            self._mirrored = False
        elif self._position.x() + bounds.right() >= parent_width:
            self._position.setX(parent_width - bounds.right())
            self._walk_direction = -1
            self._walk_velocity = -abs(self._walk_velocity)
            self._mirrored = True

    def _advance_flight(self, delta_seconds: float) -> None:
        if self._position is None:
            return
        if self._angle_resetting:
            response = 1.0 - math.exp(-self.ANGLE_RESET_RATE * delta_seconds)
            self._angle += (0.0 - self._angle) * response
            if abs(self._angle) <= 0.8:
                self._angle = 0.0
                previous = self.state
                self._machine.flight_stopped()
                self._after_state_change(previous)
            return

        self._position = QPointF(
            self._position.x() + self._velocity.x() * delta_seconds,
            self._position.y() + self._velocity.y() * delta_seconds,
        )
        self._angle = self._normalized_angle(
            self._angle + self._angular_velocity * delta_seconds
        )

        speed_decay = math.exp(-self.LINEAR_DRAG * delta_seconds)
        self._velocity = QPointF(
            self._velocity.x() * speed_decay,
            self._velocity.y() * speed_decay,
        )
        self._angular_velocity *= math.exp(-self.ANGULAR_DRAG * delta_seconds)
        self._handle_flying_collisions()

        speed = math.hypot(self._velocity.x(), self._velocity.y())
        if speed <= self.STOP_SPEED:
            self._velocity = QPointF()
            self._angular_velocity = 0.0
            self._angle = self._normalized_angle(self._angle)
            self._angle_resetting = True

    def _handle_flying_collisions(self) -> None:
        if self._position is None:
            return
        bounds = self._visible_local_bounds()
        parent = self.parentWidget()
        contacts: list[tuple[QPointF, str, float]] = []

        if self._position.x() + bounds.left() < 0:
            self._position.setX(-bounds.left())
            self._velocity.setX(abs(self._velocity.x()) * self.BOUNCE_RESTITUTION)
            contacts.append((QPointF(0, self._position.y()), "vertical", 90.0))
        elif self._position.x() + bounds.right() > parent.width():
            self._position.setX(parent.width() - bounds.right())
            self._velocity.setX(-abs(self._velocity.x()) * self.BOUNCE_RESTITUTION)
            contacts.append(
                (QPointF(parent.width(), self._position.y()), "vertical", -90.0)
            )

        if self._position.y() + bounds.top() < 0:
            self._position.setY(-bounds.top())
            self._velocity.setY(abs(self._velocity.y()) * self.BOUNCE_RESTITUTION)
            contacts.append((QPointF(self._position.x(), 0), "horizontal", 180.0))
        elif self._position.y() + bounds.bottom() > parent.height():
            self._position.setY(parent.height() - bounds.bottom())
            self._velocity.setY(-abs(self._velocity.y()) * self.BOUNCE_RESTITUTION)
            contacts.append(
                (QPointF(self._position.x(), parent.height()), "horizontal", 0.0)
            )

        for contact, axis, hit_rotation in contacts:
            self._machine.wall_hit()
            tangent = (
                self._velocity.y()
                if axis == "vertical"
                else self._velocity.x()
            )
            self._angular_velocity = self._clamp(
                -self._angular_velocity * 0.76 + tangent * 0.12,
                -self.MAX_ANGULAR_SPEED,
                self.MAX_ANGULAR_SPEED,
            )
            self._show_hit(contact, hit_rotation)
            self.wall_hit.emit(contact)

    def _advance_frame(self, milliseconds: float) -> None:
        if not self._animations_enabled or self.state == TsunaoState.FLYING:
            return
        frames = self._frames.get(self.state, ())
        if len(frames) < 2:
            return
        interval = self._FRAME_INTERVALS_MS.get(self.state, 520)
        self._frame_elapsed_ms += milliseconds
        while self._frame_elapsed_ms >= interval:
            self._frame_elapsed_ms -= interval
            self._frame_index = (self._frame_index + 1) % len(frames)

    def _after_state_change(self, previous: TsunaoState) -> None:
        if self.state == previous:
            return
        self._frame_index = 0
        self._frame_elapsed_ms = 0.0
        if self.state == TsunaoState.WALKING:
            self._walk_direction = self._random.choice((-1, 1))
            self._walk_velocity = 0.0
        else:
            self._walk_velocity = 0.0
        self._set_orientation_for_state()
        if self.state != TsunaoState.FLYING:
            self._angle = 0.0
            self._angular_velocity = 0.0
            self._angle_resetting = False
        self.setAccessibleName(MASCOT_ACCESSIBLE_NAMES[self.state])
        self.state_changed.emit(self.state.value)
        self.update()

    def _set_orientation_for_state(self) -> None:
        if self.state == TsunaoState.WALKING:
            self._mirrored = self._walk_direction < 0
        elif self.state in RANDOM_FLIP_STATES:
            self._mirrored = bool(self._random.getrandbits(1))
        else:
            self._mirrored = False

    def _current_pixmap(self) -> QPixmap:
        frames = self._frames.get(self.state, ())
        if not frames:
            return QPixmap()
        index = (
            self._machine.flying_frame
            if self.state == TsunaoState.FLYING
            else self._frame_index
        )
        return frames[index % len(frames)]

    def _visible_local_bounds(self) -> QRectF:
        pixmap = self._current_pixmap()
        if pixmap.isNull():
            half = self.DISPLAY_SIZE / 2
            return QRectF(-half, -half, self.DISPLAY_SIZE, self.DISPLAY_SIZE)
        unrotated = self._frame_bounds.get(
            pixmap.cacheKey(),
            QRectF(
                -self.DISPLAY_SIZE / 2,
                -self.DISPLAY_SIZE / 2,
                self.DISPLAY_SIZE,
                self.DISPLAY_SIZE,
            ),
        )
        left = unrotated.left()
        top = unrotated.top()
        right = unrotated.right()
        bottom = unrotated.bottom()
        if self._mirrored:
            left, right = -right, -left

        radians = math.radians(self._angle)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        corners = ((left, top), (right, top), (right, bottom), (left, bottom))
        transformed = [
            (x * cosine - y * sine, x * sine + y * cosine)
            for x, y in corners
        ]
        xs = [point[0] for point in transformed]
        ys = [point[1] for point in transformed]
        return QRectF(
            min(xs),
            min(ys),
            max(xs) - min(xs),
            max(ys) - min(ys),
        )

    def _show_hit(self, contact: QPointF, rotation: float) -> None:
        if self._user_hidden or self._hit_pixmap.isNull():
            return
        self._hit_effect.show_effect(contact, rotation)
        self.raise_()

    def _sampled_pointer_velocity(self) -> QPointF:
        if len(self._drag_samples) < 2:
            return QPointF()
        newest_time, newest_point = self._drag_samples[-1]
        oldest_time, oldest_point = self._drag_samples[0]
        for sample_time, sample_point in self._drag_samples:
            if newest_time - sample_time <= 0.11:
                oldest_time, oldest_point = sample_time, sample_point
                break
        elapsed = max(0.001, newest_time - oldest_time)
        return QPointF(
            (newest_point.x() - oldest_point.x()) / elapsed,
            (newest_point.y() - oldest_point.y()) / elapsed,
        )

    def _cursor_target(self, point: QPointF) -> QPointF:
        parent = self.parentWidget()
        return QPointF(
            self._clamp(point.x(), 0.0, float(parent.width())),
            self._clamp(point.y(), 0.0, float(parent.height())),
        )

    def _ensure_initial_position(self) -> None:
        if self._position is not None:
            return
        parent = self.parentWidget()
        self._position = QPointF(
            min(float(parent.width()) * 0.16, 118.0),
            max(90.0, float(parent.height()) - 118.0),
        )
        self._keep_visible_inside_parent()

    def _keep_visible_inside_parent(self) -> None:
        if self._position is None:
            return
        bounds = self._visible_local_bounds()
        parent = self.parentWidget()
        minimum_x = -bounds.left()
        maximum_x = parent.width() - bounds.right()
        minimum_y = -bounds.top()
        maximum_y = parent.height() - bounds.bottom()
        self._position.setX(
            self._clamp(self._position.x(), minimum_x, max(minimum_x, maximum_x))
        )
        self._position.setY(
            self._clamp(self._position.y(), minimum_y, max(minimum_y, maximum_y))
        )

    def _sync_geometry(self) -> None:
        if self._position is None:
            return
        self.move(
            round(self._position.x() - self.width() / 2),
            round(self._position.y() - self.height() / 2),
        )
        self.raise_()

    @staticmethod
    def _normalized_angle(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))
