const video = document.getElementById("video");
const canvas = document.getElementById("photo");
const captureBtn = document.getElementById("capture-btn");

const cameraError = document.getElementById("camera-error");
navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    cameraError.style.display = "none";
  })
  .catch((err) => {
    video.style.display = "none";
    cameraError.textContent = `Error: ${err.name}. Please make sure you have granted permission to use the camera.`;
    cameraError.style.display = "block";
  });

captureBtn.addEventListener("click", async () => {
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.style.display = "block";

  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "capture.png");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log(data);
      alert("Server response: " + data.result);
    } catch (err) {
      alert("Error sending image to server.");
    }
  }, "image/png");
});
