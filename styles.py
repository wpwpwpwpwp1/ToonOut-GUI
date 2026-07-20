"""ToonOut의 밝은 애니메이션 작업대 테마."""


APP_STYLESHEET = """
QWidget#root {
    background: #f5f7fb;
    color: #1e2430;
    font-family: "Segoe UI", "Malgun Gothic";
    font-size: 14px;
}
QLabel#appTitle {
    color: #20263a;
    font-size: 42px;
    font-weight: 700;
}
QLabel#dialogTitle {
    color: #20263a;
    font-size: 21px;
    font-weight: 700;
}
QLabel#fieldLabel {
    color: #434b5d;
    font-size: 12px;
    font-weight: 650;
}
QLabel#storageSpace {
    color: #3f43bc;
    font-size: 13px;
    font-weight: 600;
}
QLabel#modelInstalledLabel {
    color: #18794e;
    background: #eaf8f1;
    border: 1px solid #b9e3ce;
    border-radius: 9px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#modelRequiredLabel {
    color: #b9384b;
    background: #fff1f3;
    border: 1px solid #f1c8cf;
    border-radius: 9px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#accelerationActiveLabel {
    color: #18794e;
    background: #eaf8f1;
    border: 1px solid #b9e3ce;
    border-radius: 9px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#accelerationInstalledLabel, QLabel#accelerationAvailableLabel {
    color: #3f43bc;
    background: #e9eaff;
    border: 1px solid #c3c6fa;
    border-radius: 9px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#accelerationWarningLabel {
    color: #8a5b12;
    background: #fff8e6;
    border: 1px solid #ecd59b;
    border-radius: 9px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#accelerationCpuLabel {
    color: #59647a;
    background: #f0f2f6;
    border: 1px solid #d6dae4;
    border-radius: 9px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#mutedText {
    color: #6e7687;
}
QLabel#sectionTitle, QLabel#previewTitle {
    color: #252b3a;
    font-size: 16px;
    font-weight: 650;
}
QLabel#privacyText {
    color: #59647a;
    font-size: 13px;
}
QLabel#errorText {
    color: #b9384b;
    background: #fff1f3;
    border: 1px solid #f1c8cf;
    border-radius: 8px;
    padding: 8px 10px;
}
QFrame#previewPanel, QFrame#actionBar {
    background: #ffffff;
    border: 1px solid #dde1ea;
    border-radius: 14px;
}
QFrame#storageCard {
    background: #f7f7ff;
    border: 1px solid #d8daf8;
    border-radius: 12px;
}
QFrame#guideCard {
    background: #f2f5ff;
    border: 1px solid #d7def7;
    border-radius: 12px;
}
QLabel#guideTitle {
    color: #343b74;
    font-size: 13px;
    font-weight: 700;
}
QLabel#namingPreview {
    color: #343b74;
    font-size: 15px;
    font-weight: 700;
}
QListWidget#fileList {
    background: #ffffff;
    border: 1px solid #dde1ea;
    border-radius: 14px;
    padding: 8px;
    outline: none;
}
QListWidget#fileList::item {
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 7px;
}
QListWidget#fileList::item:hover {
    background: #f4f5ff;
    border-color: #d9dbfa;
}
QListWidget#fileList::item:selected {
    background: #e9eaff;
    border-color: #8c91f5;
    color: #292f74;
}
QListWidget#fileList QScrollBar:vertical {
    width: 12px;
    margin: 5px 2px;
    background: transparent;
}
QListWidget#fileList QScrollBar::handle:vertical {
    min-height: 36px;
    background: #c5c9d6;
    border-radius: 4px;
}
QListWidget#fileList QScrollBar::handle:vertical:hover {
    background: #858bf4;
}
QListWidget#fileList QScrollBar::add-line:vertical,
QListWidget#fileList QScrollBar::sub-line:vertical {
    height: 0;
}
QListWidget#fileList QScrollBar::add-page:vertical,
QListWidget#fileList QScrollBar::sub-page:vertical {
    background: transparent;
}
QRubberBand {
    background: rgba(91, 95, 239, 38);
    border: 1px solid #696ee9;
}
QPushButton {
    min-height: 34px;
    border-radius: 9px;
    padding: 0 14px;
    font-weight: 600;
}
QPushButton#primaryButton {
    color: #ffffff;
    background: #5b5fef;
    border: 1px solid #5b5fef;
}
QPushButton#primaryButton:hover {
    background: #4d51dc;
}
QPushButton#primaryButton:disabled {
    color: #f0f1f5;
    background: #aeb2c7;
    border-color: #aeb2c7;
}
QPushButton#secondaryButton, QPushButton#quietButton,
QPushButton#modeButton {
    color: #343b4d;
    background: #ffffff;
    border: 1px solid #d6dae4;
}
QPushButton#secondaryButton:hover, QPushButton#quietButton:hover,
QPushButton#modeButton:hover {
    border-color: #858bf4;
    background: #f7f7ff;
}
QPushButton#modeButton:checked {
    color: #3f43bc;
    background: #e9eaff;
    border-color: #8c91f5;
}
QCheckBox#mascotToggle {
    min-height: 34px;
    padding: 0 12px 0 10px;
    spacing: 7px;
    color: #4d5265;
    background: #ffffff;
    border: 1px solid #d6dae4;
    border-radius: 9px;
    font-weight: 600;
}
QCheckBox#mascotToggle:hover {
    color: #34385d;
    background: #f7f7ff;
    border-color: #9da1ed;
}
QCheckBox#mascotToggle:checked {
    color: #3f43bc;
    background: #e9eaff;
    border-color: #8c91f5;
}
QCheckBox#mascotToggle:focus {
    padding: 0 11px 0 9px;
    border: 2px solid #696ee9;
}
QPushButton#dangerButton {
    color: #b9384b;
    background: #fff4f5;
    border: 1px solid #efc5cc;
}
QPushButton#modelInstalledButton {
    color: #18794e;
    background: #eaf8f1;
    border: 1px solid #b9e3ce;
}
QPushButton#modelInstalledButton:hover {
    background: #dcf3e7;
    border-color: #79c8a2;
}
QPushButton#modelRequiredButton {
    color: #b9384b;
    background: #fff1f3;
    border: 1px solid #f1c8cf;
}
QPushButton#modelRequiredButton:hover {
    background: #ffe5e9;
    border-color: #de8e9b;
}
QPushButton#modelBusyButton {
    color: #8a5b12;
    background: #fff8e6;
    border: 1px solid #ecd59b;
}
QPushButton#accelerationActiveButton {
    color: #18794e;
    background: #eaf8f1;
    border: 1px solid #b9e3ce;
}
QPushButton#accelerationInstalledButton,
QPushButton#accelerationAvailableButton {
    color: #3f43bc;
    background: #e9eaff;
    border: 1px solid #c3c6fa;
}
QPushButton#accelerationWarningButton {
    color: #8a5b12;
    background: #fff8e6;
    border: 1px solid #ecd59b;
}
QPushButton#accelerationCpuButton,
QPushButton#accelerationCheckingButton {
    color: #59647a;
    background: #f0f2f6;
    border: 1px solid #d6dae4;
}
QPushButton#updateCheckingButton {
    color: #59647a;
    background: #f0f2f6;
    border: 1px solid #d6dae4;
}
QPushButton#updateDownloadingButton {
    color: #3f43bc;
    background: #e9eaff;
    border: 1px solid #c3c6fa;
}
QPushButton#updateReadyButton {
    color: #18794e;
    background: #eaf8f1;
    border: 1px solid #b9e3ce;
}
QPushButton#updateWarningButton {
    color: #8a5b12;
    background: #fff8e6;
    border: 1px solid #ecd59b;
}
QPushButton#accelerationActiveButton:hover,
QPushButton#accelerationInstalledButton:hover,
QPushButton#accelerationAvailableButton:hover,
QPushButton#accelerationWarningButton:hover,
QPushButton#accelerationCpuButton:hover {
    border-color: #858bf4;
    background: #f7f7ff;
}
QPushButton#updateReadyButton:hover,
QPushButton#updateWarningButton:hover {
    border-color: #858bf4;
    background: #f7f7ff;
}
QComboBox, QSpinBox {
    min-height: 34px;
    color: #343b4d;
    background: #ffffff;
    border: 1px solid #d6dae4;
    border-radius: 9px;
    padding: 0 28px 0 10px;
}
QComboBox:hover {
    background: #fafaff;
    border-color: #9ba0ee;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 30px;
    border: none;
    border-left: 1px solid #eceef4;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
}
QListView#comboPopup {
    color: #252b3a;
    background: #ffffff;
    border: none;
    outline: none;
    padding: 5px;
}
QFrame#comboPopupContainer {
    background: #ffffff;
    border: 1px solid #cfd3df;
    border-radius: 10px;
}
QPushButton:focus, QComboBox:focus, QSpinBox:focus, QLineEdit:focus {
    border: 2px solid #5b5fef;
}
QLineEdit#pathField, QLineEdit#namingField {
    min-height: 34px;
    color: #343b4d;
    background: #ffffff;
    border: 1px solid #d6dae4;
    border-radius: 9px;
    padding: 0 10px;
}
QProgressBar {
    min-height: 12px;
    max-height: 12px;
    border: none;
    border-radius: 6px;
    background: #e7e9f0;
    text-align: center;
}
QProgressBar::chunk {
    border-radius: 6px;
    background: #5b5fef;
}
QSplitter::handle:horizontal {
    background: #e2e5ee;
    width: 10px;
    margin: 12px 3px;
    border-radius: 2px;
}
QSplitter::handle:horizontal:hover,
QSplitter::handle:horizontal:focus {
    background: #7d82ef;
}
"""
