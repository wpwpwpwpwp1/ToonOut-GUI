import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QWidget

from mascot import (
    MASCOT_ASSET_FILES,
    TsunaoJobState,
    TsunaoState,
    TsunaoStateMachine,
    TsunaoWidget,
)
from scripts.prepare_mascot_assets import (
    OUTPUT_SIZE,
    SOURCE_TO_OUTPUT,
    STATE_CONTENT_EXTENT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASCOT_DIRECTORY = PROJECT_ROOT / "assets" / "mascot"


class MascotAssetTests(unittest.TestCase):
    def test_every_declared_source_file_has_one_runtime_role(self):
        self.assertEqual(len(SOURCE_TO_OUTPUT), 30)
        self.assertEqual(len(set(SOURCE_TO_OUTPUT)), 30)
        self.assertEqual(len(set(SOURCE_TO_OUTPUT.values())), 30)

    def test_every_source_role_has_an_optimized_rgba_asset(self):
        filenames = {
            filename
            for state_files in MASCOT_ASSET_FILES.values()
            for filename in state_files
        } | {"icon.png", "hit.png"}

        self.assertEqual(len(filenames), 30)
        for filename in filenames:
            with self.subTest(filename=filename):
                with Image.open(MASCOT_DIRECTORY / filename) as image:
                    self.assertEqual(image.size, (640, 640))
                    self.assertEqual(image.mode, "RGBA")

    def test_visible_frames_share_one_scale_within_each_state(self):
        def visible_extent(path):
            with Image.open(path) as image:
                bounds = image.getchannel("A").getbbox()
                self.assertIsNotNone(bounds)
                return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        for state, filenames in MASCOT_ASSET_FILES.items():
            with self.subTest(state=state.value):
                extents = [
                    visible_extent(MASCOT_DIRECTORY / filename)
                    for filename in filenames
                ]
                self.assertEqual(max(extents), STATE_CONTENT_EXTENT)

    def test_crouching_frames_are_shorter_but_keep_the_same_floor(self):
        def visible_bounds(path):
            with Image.open(path) as image:
                bounds = image.getchannel("A").getbbox()
                self.assertIsNotNone(bounds)
                return bounds

        expected_floor = round(OUTPUT_SIZE / 2 + STATE_CONTENT_EXTENT / 2)
        cases = (
            ("standing-1.png", "standing-2.png"),
            ("angry-1.png", "angry-2.png"),
        )
        for output_first, output_second in cases:
            with self.subTest(second=output_second):
                first = visible_bounds(MASCOT_DIRECTORY / output_first)
                second = visible_bounds(MASCOT_DIRECTORY / output_second)
                first_extent = max(
                    first[2] - first[0], first[3] - first[1]
                )
                second_extent = max(
                    second[2] - second[0], second[3] - second[1]
                )

                self.assertEqual(first_extent, STATE_CONTENT_EXTENT)
                self.assertLess(second_extent, first_extent)
                self.assertEqual(first[3], expected_floor)
                self.assertEqual(second[3], expected_floor)


class MascotStateMachineTests(unittest.TestCase):
    def test_begin_and_default_routines_follow_the_requested_timing(self):
        class MinimumWalkingRandom:
            @staticmethod
            def uniform(minimum, maximum):
                del maximum
                return minimum

        machine = TsunaoStateMachine(random_source=MinimumWalkingRandom())

        machine.advance(60_000)
        self.assertEqual(machine.state, TsunaoState.SLEEPING)
        self.assertTrue(machine.image_loaded())
        self.assertFalse(machine.image_loaded())

        machine.advance(4_999)
        self.assertEqual(machine.state, TsunaoState.AWAKE)
        machine.advance(1)
        self.assertEqual(machine.state, TsunaoState.STANDING)

        machine.advance(3_000)
        self.assertEqual(machine.state, TsunaoState.WALKING)
        machine.advance(5_000)
        self.assertEqual(machine.state, TsunaoState.SMUG)
        machine.advance(5_000)
        self.assertEqual(machine.state, TsunaoState.WALKING)
        machine.advance(5_000)
        self.assertEqual(machine.state, TsunaoState.TOUCHEAR)
        machine.advance(3_000)
        self.assertEqual(machine.state, TsunaoState.WALKING)

        machine.advance(5_000)
        self.assertEqual(machine.state, TsunaoState.SMUG)
        machine.advance(5_000)
        self.assertEqual(machine.state, TsunaoState.WALKING)
        machine.advance(5_000)
        self.assertEqual(machine.state, TsunaoState.TOUCHEAR)
        machine.advance(3_000)
        self.assertEqual(machine.state, TsunaoState.DROWSY)
        machine.advance(10_000)
        self.assertEqual(machine.state, TsunaoState.SLEEPING)

    def test_slow_release_gets_angry_then_resumes_the_saved_routine(self):
        machine = TsunaoStateMachine()
        machine.image_loaded()
        machine.advance(5_000)
        machine.advance(1_200)

        machine.pick_up()
        machine.release(fast=False)
        self.assertEqual(machine.state, TsunaoState.ANGRY)
        machine.advance(2_999)
        self.assertEqual(machine.state, TsunaoState.ANGRY)
        machine.advance(1)
        self.assertEqual(machine.state, TsunaoState.STANDING)
        self.assertEqual(machine.elapsed_ms, 1_200)

    def test_flight_can_be_regrabbed_and_defers_processing_pose_changes(self):
        machine = TsunaoStateMachine()
        machine.image_loaded()
        machine.pick_up()
        machine.release(fast=True)
        self.assertEqual(machine.state, TsunaoState.FLYING)

        machine.wall_hit()
        self.assertEqual(machine.flying_frame, 1)
        machine.pick_up()
        self.assertEqual(machine.state, TsunaoState.PICKED)
        machine.release(fast=True)
        self.assertEqual(machine.flying_frame, 0)

        machine.processing_started()
        machine.processing_finished(completed=True)
        self.assertEqual(machine.job_state, TsunaoJobState.COMPLETE)
        self.assertEqual(machine.state, TsunaoState.FLYING)

        machine.flight_stopped()
        self.assertEqual(machine.state, TsunaoState.HURT)
        machine.advance(4_999)
        self.assertEqual(machine.state, TsunaoState.HURT)
        machine.advance(1)
        self.assertEqual(machine.state, TsunaoState.COMPLETE)
        machine.advance(5_000)
        self.assertEqual(machine.state, TsunaoState.STANDING)

    def test_sleeping_throw_returns_to_default_after_hurt(self):
        machine = TsunaoStateMachine()
        machine.pick_up()
        machine.release(fast=True)
        machine.flight_stopped()

        machine.advance(5_000)

        self.assertEqual(machine.state, TsunaoState.STANDING)

    def test_each_walking_phase_randomly_lasts_between_five_and_ten_seconds(self):
        class WalkingDurationRandom:
            def __init__(self):
                self._durations = iter((5_000, 10_000))

            def uniform(self, minimum, maximum):
                duration = next(self._durations)
                self.assert_duration_in_bounds(duration, minimum, maximum)
                return duration

            @staticmethod
            def assert_duration_in_bounds(duration, minimum, maximum):
                if not minimum <= duration <= maximum:
                    raise AssertionError("walking duration is outside its bounds")

        machine = TsunaoStateMachine(
            random_source=WalkingDurationRandom(),
        )
        machine.image_loaded()
        machine.advance(5_000)
        machine.advance(3_000)

        machine.advance(4_999)
        self.assertEqual(machine.state, TsunaoState.WALKING)
        machine.advance(1)
        self.assertEqual(machine.state, TsunaoState.SMUG)
        machine.advance(5_000)

        machine.advance(9_999)
        self.assertEqual(machine.state, TsunaoState.WALKING)
        machine.advance(1)
        self.assertEqual(machine.state, TsunaoState.TOUCHEAR)


class MascotWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_widget_loads_all_frames_and_anchors_hit_at_bottom_center(self):
        parent = QWidget()
        parent.resize(900, 620)
        widget = TsunaoWidget(parent, animations_enabled=False)
        widget.set_assets(MASCOT_DIRECTORY)
        parent.show()
        self.app.processEvents()

        self.assertEqual(widget.state, TsunaoState.SLEEPING)
        self.assertEqual(widget.accessibleName(), "잠든 츠나오")
        self.assertEqual(set(widget._frames), set(TsunaoState))

        widget._show_hit(QPointF(160, 210), 0.0)
        self.assertEqual(
            widget._hit_effect.geometry().x()
            + widget._hit_effect.width() // 2,
            160,
        )
        self.assertEqual(
            widget._hit_effect.geometry().y()
            + widget._hit_effect.height() // 2,
            210,
        )
        self.assertEqual(widget._hit_effect._rotation, 0.0)
        self.assertEqual(widget._hit_effect.opacity, 1.0)

        widget.begin_pick(widget.position)
        self.assertEqual(widget.state, TsunaoState.PICKED)
        widget.release_pick(widget.position)
        self.assertEqual(widget.state, TsunaoState.ANGRY)
        widget.shutdown()
        parent.close()

    def test_user_hidden_setting_hides_mascot_and_hit_effect_together(self):
        parent = QWidget()
        parent.resize(900, 620)
        widget = TsunaoWidget(parent, animations_enabled=False)
        widget.set_assets(MASCOT_DIRECTORY)
        parent.show()
        self.app.processEvents()

        widget._show_hit(QPointF(160, 210), 0.0)
        self.assertFalse(widget._hit_effect.isHidden())

        widget.set_user_hidden(True)
        self.assertTrue(widget.isHidden())
        self.assertTrue(widget._hit_effect.isHidden())

        widget.set_user_hidden(False)
        self.assertFalse(widget.isHidden())
        widget.shutdown()
        parent.close()

    def test_drag_target_is_centered_and_follows_with_a_short_delay(self):
        parent = QWidget()
        parent.resize(900, 620)
        widget = TsunaoWidget(parent, animations_enabled=False)
        widget.set_assets(MASCOT_DIRECTORY)
        widget.place_initial(QPointF(120, 420))

        widget.begin_pick(QPointF(320, 180))
        before = widget.position
        widget._advance_drag(0.05)
        after = widget.position

        self.assertEqual(widget._drag_target, QPointF(320, 180))
        self.assertGreater(after.x(), before.x())
        self.assertLess(after.x(), 320)
        self.assertLess(after.y(), before.y())
        self.assertGreater(after.y(), 180)
        self.assertGreater(after.x(), 300)
        self.assertLess(after.y(), 200)
        widget.release_pick(QPointF(320, 180))
        widget.shutdown()
        parent.close()

    def test_flying_bounces_changes_frame_and_can_be_picked_again(self):
        parent = QWidget()
        parent.resize(420, 320)
        widget = TsunaoWidget(parent, animations_enabled=False)
        widget.set_assets(MASCOT_DIRECTORY)
        widget.place_initial(QPointF(360, 160))
        widget._machine.pick_up()
        widget._machine.release(fast=True)
        widget._velocity = QPointF(700, 0)
        widget._angular_velocity = 240

        widget._advance_flight(0.08)

        self.assertEqual(widget.state, TsunaoState.FLYING)
        self.assertEqual(widget._machine.flying_frame, 1)
        self.assertLess(widget._velocity.x(), 0)
        self.assertFalse(widget._hit_effect.isHidden())
        self.assertEqual(widget._hit_effect._rotation, -90.0)

        widget.begin_pick(widget.position)
        self.assertEqual(widget.state, TsunaoState.PICKED)
        widget.release_pick(widget.position)
        widget.shutdown()
        parent.close()

    def test_fast_pointer_release_enters_flying_with_bounded_velocity(self):
        parent = QWidget()
        parent.resize(900, 620)
        widget = TsunaoWidget(parent, animations_enabled=False)
        widget.set_assets(MASCOT_DIRECTORY)
        widget.place_initial(QPointF(120, 420))

        with patch("mascot.time.monotonic", side_effect=(1.0, 1.1)):
            widget.begin_pick(QPointF(120, 420))
            widget.release_pick(QPointF(360, 420))

        speed = (widget._velocity.x() ** 2 + widget._velocity.y() ** 2) ** 0.5
        self.assertEqual(widget.state, TsunaoState.FLYING)
        self.assertLessEqual(speed, widget.MAX_THROW_SPEED)
        self.assertGreater(speed, widget.THROW_SPEED_THRESHOLD)
        self.assertGreater(speed, 4_000)
        widget.shutdown()
        parent.close()

    def test_stopped_flight_quickly_returns_to_zero_before_hurt(self):
        parent = QWidget()
        parent.resize(420, 320)
        widget = TsunaoWidget(parent, animations_enabled=False)
        widget.set_assets(MASCOT_DIRECTORY)
        widget.place_initial(QPointF(210, 160))
        widget._machine.pick_up()
        widget._machine.release(fast=True)
        widget._velocity = QPointF()
        widget._angle = 95.0

        widget._advance_flight(0.016)
        self.assertEqual(widget.state, TsunaoState.FLYING)
        self.assertTrue(widget._angle_resetting)
        for _ in range(30):
            if widget.state != TsunaoState.FLYING:
                break
            widget._advance_flight(0.016)

        self.assertEqual(widget.state, TsunaoState.HURT)
        self.assertEqual(widget._angle, 0.0)
        widget.shutdown()
        parent.close()

    def test_hit_effect_rotation_matches_each_wall_and_fades(self):
        parent = QWidget()
        parent.resize(500, 400)
        widget = TsunaoWidget(parent, animations_enabled=True)
        widget.set_assets(MASCOT_DIRECTORY)
        widget._machine.pick_up()
        widget._machine.release(fast=True)
        widget._angle = 0.0

        collisions = (
            (QPointF(-100, 200), QPointF(-600, 0), 90.0),
            (QPointF(600, 200), QPointF(600, 0), -90.0),
            (QPointF(250, -100), QPointF(0, -600), 180.0),
            (QPointF(250, 500), QPointF(0, 600), 0.0),
        )
        for position, velocity, expected_rotation in collisions:
            widget._position = position
            widget._velocity = velocity
            widget._handle_flying_collisions()
            self.assertEqual(widget._hit_effect._rotation, expected_rotation)

        widget._hit_effect._sequence.setCurrentTime(40)
        fade_in_opacity = widget._hit_effect.opacity
        widget._hit_effect._sequence.setCurrentTime(250)
        fade_out_opacity = widget._hit_effect.opacity
        self.assertGreater(fade_in_opacity, 0.0)
        self.assertGreater(fade_out_opacity, 0.0)
        self.assertLess(fade_out_opacity, fade_in_opacity)
        widget.shutdown()
        parent.close()

    def test_random_flip_is_shared_by_both_frames_of_a_state(self):
        class AlwaysFlipRandom:
            @staticmethod
            def choice(values):
                return values[-1]

            @staticmethod
            def getrandbits(bits):
                del bits
                return 1

        parent = QWidget()
        parent.resize(900, 620)
        widget = TsunaoWidget(
            parent,
            animations_enabled=True,
            random_source=AlwaysFlipRandom(),
        )
        widget.set_assets(MASCOT_DIRECTORY)

        self.assertTrue(widget.mirrored)
        widget._advance_frame(950)
        self.assertEqual(widget._frame_index, 1)
        self.assertTrue(widget.mirrored)

        widget.image_loaded()
        self.assertTrue(widget.mirrored)
        widget._advance_frame(520)
        self.assertEqual(widget._frame_index, 1)
        self.assertTrue(widget.mirrored)
        widget.shutdown()
        parent.close()

    def test_walking_reaches_the_higher_speed_with_strong_acceleration(self):
        parent = QWidget()
        parent.resize(900, 620)
        widget = TsunaoWidget(parent, animations_enabled=True)
        widget.set_assets(MASCOT_DIRECTORY)
        widget.place_initial(QPointF(450, 310))
        widget.image_loaded()
        widget.advance_time(5_000)
        widget.advance_time(3_000)
        before_x = widget.position.x()

        widget._advance_walk(0.2)

        self.assertEqual(abs(widget._walk_velocity), widget.WALK_SPEED)
        self.assertGreater(abs(widget.position.x() - before_x), 15)
        widget.shutdown()
        parent.close()


if __name__ == "__main__":
    unittest.main()
