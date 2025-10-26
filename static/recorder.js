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
            player.src = audioURL;
            player.load();
            
            const file = new File([blob], "recorded.webm", { type: "audio/webm" });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);

            const input = document.getElementById('blob');
            input.files = dataTransfer.files;

            document.getElementById("recordForm").style.display = "block";
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