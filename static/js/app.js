// static/js/app.js
document.addEventListener('DOMContentLoaded', () => {
    const attendanceForm = document.querySelector('form[action="/attendance"]');
    const leavingForm = document.querySelector('form[action="/leaving"]');
    const buttonGroup = document.querySelector('.button-group');

    // オーバーレイ要素を動的に作成
    const overlay = document.createElement('div');
    overlay.id = 'overlay';
    document.body.appendChild(overlay);

    function disableButtonsAndShowOverlay() {
        const buttons = buttonGroup.querySelectorAll('button');
        buttons.forEach(button => {
            button.disabled = true;
            button.classList.add('disabled');
        });
        overlay.style.display = 'block';
    }

    if (attendanceForm) {
        attendanceForm.addEventListener('submit', (event) => {
            disableButtonsAndShowOverlay();
        });
    }

    if (leavingForm) {
        leavingForm.addEventListener('submit', (event) => {
            disableButtonsAndShowOverlay();
        });
    }
});