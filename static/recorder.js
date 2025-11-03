let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
        
        mediaRecorder.start();
        audioChunks = [];

        document.getElementById("stopBtn").disabled = false;

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
            const audioURL = URL.createObjectURL(blob);
            const player = document.getElementById('player');
            if (player) {
                player.src = audioURL;
                player.classList.remove('hidden');
                player.load();
            }
            
            const file = new File([blob], "recorded.webm", { type: "audio/webm" });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);

            const input = document.getElementById('blob');
            input.files = dataTransfer.files;

            const form = document.getElementById("recordForm");
            if (form) {
                form.style.display = "block";
            }

            // Route the main Detect button to submit the recorded-audio form
            const detectBtn = document.getElementById('detectBtn');
            if (detectBtn) {
                detectBtn.setAttribute('form', 'recordForm');
            }
        };
    });
}

function stopRecording() {
    mediaRecorder.stop();
    document.getElementById("stopBtn").disabled = true;
}

window.addEventListener("DOMContentLoaded", () => {
    const forms = document.querySelectorAll("form");
    const loading = document.getElementById("loading");

    forms.forEach(form => {
        form.addEventListener("submit", () => {
            loading.classList.remove("hidden");
        });
    });
});
