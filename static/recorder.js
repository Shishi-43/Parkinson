let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        

        
        mediaRecorder.start();
        audioChunks = [];

        document.getElementById("stopBtn").disabled = false;

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            const audioURL = URL.createObjectURL(blob);
            document.getElementById('player').src = audioURL;

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
