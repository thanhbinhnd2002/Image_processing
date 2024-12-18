import React, { useState } from "react";
import axios from "axios";

const TrainingForm = () => {
    const [params, setParams] = useState({ epochs: 20, batchSize: 8, lr: 0.001, conf_threshold: 0.3 });
    const [message, setMessage] = useState(""); // Status for backend responses
    const [status, setStatus] = useState(""); // Status for detection state
    const [image, setImage] = useState(null); // Input image
    const [resultImage, setResultImage] = useState(null); // Result image

    // Handle parameter input changes
    const handleParamChange = (e) => {
        const { name, value } = e.target;
        setParams({ ...params, [name]: value });
    };

    // Handle file input change


    // Start training
    const startTraining = async () => {
        setMessage("Detecting..."); // Show detecting status
        try {
            const response = await axios.post("http://localhost:5000/start-training", params);
            setMessage(response.data.status);
        } catch (error) {
            setMessage("Error starting training");
            console.error(error);
        }
    };

    return (
        <div>
            <h2>Object Detection System</h2>

            {/* Training Parameters */}
            <div>
                <h3>Training Parameters</h3>
                <div style={{ marginBottom: "10px" }}>
                    <label>
                        <strong>Number of Epochs:</strong> (The number of full passes through the training data)
                        <br />
                        <input
                            name="epochs"
                            type="number"
                            value={params.epochs}
                            onChange={handleParamChange}
                            placeholder="Epochs"
                        />
                    </label>
                </div>
                <div style={{ marginBottom: "10px" }}>
                    <label>
                        <strong>Batch Size:</strong> (Number of samples processed before updating the model)
                        <br />
                        <input
                            name="batchSize"
                            type="number"
                            value={params.batchSize}
                            onChange={handleParamChange}
                            placeholder="Batch Size"
                        />
                    </label>
                </div>
                <div style={{ marginBottom: "10px" }}>
                    <label>
                        <strong>Learning Rate:</strong> (Step size for optimizing the model weights)
                        <br />
                        <input
                            name="lr"
                            type="number"
                            step="0.0001"
                            value={params.lr}
                            onChange={handleParamChange}
                            placeholder="Learning Rate"
                        />
                    </label>
                </div>
                <button onClick={startTraining} style={{ marginTop: "10px" }}>
                    Start Training
                </button>
            </div>
        </div>
    );
};

export default TrainingForm;
