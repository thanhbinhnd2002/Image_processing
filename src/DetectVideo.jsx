import React, { useState } from "react";
import axios from "axios";

const DetectVideo = () => {
    const [params, setParams] = useState({ conf_threshold: 0.3 }); // Default confidence threshold
    const [message, setMessage] = useState(""); // Status for backend responses
    const [status, setStatus] = useState(""); // Status for detection state
    const [video, setVideo] = useState(null); // Input video
    const [resultVideo, setResultVideo] = useState(null); // Result video

    const handleParamChange = (e) => {
        const { name, value } = e.target;
        setParams({ ...params, [name]: value });
    };

    const handleFileChange = (e) => {
        setVideo(e.target.files[0]);
        setResultVideo(null); // Reset result video when a new file is selected
        setStatus(""); // Reset status when a new file is selected
    };

    const startDetection = async () => {
        if (!video) {
            alert("Please select a video.");
            return;
        }

        setStatus("Detecting..."); // Show detecting status

        const formData = new FormData();
        formData.append("file", video); // Append video file
        formData.append("conf_threshold", params.conf_threshold); // Append confidence threshold

        try {
            const response = await axios.post("http://localhost:5000/detect-video", formData, {
                headers: { "Content-Type": "multipart/form-data" },
                responseType: "blob", // Expect a blob (video file) as response
            });

            const url = URL.createObjectURL(new Blob([response.data])); // Create a URL for the result video
            setResultVideo(url); // Set the result video
            setStatus("Detection completed successfully!"); // Update status
        } catch (error) {
            setStatus("Error during detection."); // Update status on error
            console.error(error);
        }
    };

    return (
        <div>
            <div style={{ marginTop: "20px" }}>
                <h3>Video Detection</h3>
                <label>
                    Select a video:
                    <input type="file" onChange={handleFileChange} accept="video/*" />
                </label>
                <div style={{ display: "flex", marginTop: "10px", justifyContent: "center" }}>
                    <p>Confidence Threshold: </p>
                    <input
                        name="conf_threshold"
                        type="number"
                        step="0.1"
                        value={params.conf_threshold}
                        onChange={handleParamChange}
                        placeholder="Confidence Threshold"
                    />
                </div>
                <button onClick={startDetection} style={{ marginTop: "10px" }}>
                    Start Detection
                </button>
            </div>

            {/* Messages */}
            <p><strong>Detection Status:</strong> {status}</p>

            {/* Display Input Video */}
            <div style={{ display: "flex", justifyContent: "space-between", gap: 50 }}>
                <div>
                    <h3>Input Video</h3>
                    {video && (
                        <video controls style={{ maxWidth: "100%" }}>
                            <source src={URL.createObjectURL(video)} type="video/mp4" />
                            Your browser does not support the video tag.
                        </video>
                    )}
                </div>

                {/* Display Result Video */}
                <div>
                    <h3>Result Video</h3>
                    {resultVideo && (
                        <video controls style={{ maxWidth: "100%" }}>
                            <source src={resultVideo} type="video/mp4" />
                            Your browser does not support the video tag.
                        </video>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DetectVideo;
