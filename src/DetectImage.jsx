import React, {useState} from "react";
import axios from "axios";

const DetectImage = () => {
    const [params, setParams] = useState({});

    const [message, setMessage] = useState(""); // Status for backend responses
    const [status, setStatus] = useState(""); // Status for detection state
    const [image, setImage] = useState(null); // Input image
    const [resultImage, setResultImage] = useState(null); // Result image

    const handleParamChange = (e) => {
        const {name, value} = e.target;
        setParams({...params, [name]: value});
    };

    const handleFileChange = (e) => {
        setImage(e.target.files[0]);
        setResultImage(null); // Reset result image when a new file is selected
        setStatus(""); // Reset status when a new file is selected
    };
    const startDetection = async () => {
        if (!image) {
            alert("Please select an image.");
            return;
        }

        setStatus("Detecting..."); // Show detecting status

        const formData = new FormData();
        formData.append("file", image);
        formData.append("conf_threshold", params.conf_threshold); // Append confidence threshold

        try {
            const response = await axios.post("http://localhost:5000/detect-image", formData, {
                headers: {"Content-Type": "multipart/form-data"},
                responseType: "blob", // Expect a blob (image file) as response
            });

            const url = URL.createObjectURL(new Blob([response.data])); // Create a URL for the result image
            setResultImage(url); // Set the result image
            setStatus("Detection completed successfully!"); // Update status
        } catch (error) {
            setStatus("Error during detection."); // Update status on error
            console.error(error);
        }
    };
    return (
        <div>
            <div style={{marginTop: "20px"}}>
                <h3>Image Detection</h3>
                <label>
                    Select an image:
                    <input type="file" onChange={handleFileChange} accept="image/*"/>
                </label>
                <div style={{display: "flex", marginTop: "10px", justifyContent: 'center'}}>
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
                <button onClick={startDetection} style={{marginTop: "10px"}}>
                    Start Detection
                </button>
            </div>

            {/* Messages */
            }
            <p><strong>Training Status:</strong> {message}</p>
            <p><strong>Detection Status:</strong> {status}</p>

            {/* Display Input Image */
            }
            <div style={{display: 'flex', justifyContent: 'space-between', gap: 50}}>
                <div>
                    <h3>Input Image</h3>
                    {image && <img src={URL.createObjectURL(image)} alt="Input" style={{maxWidth: "100%"}}/>}
                </div>

                {/* Display Result Image */
                }
                <div>
                    <h3>Result Image</h3>
                    {resultImage && <img src={resultImage} alt="Result" style={{maxWidth: "100%"}}/>}
                </div>
            </div>
        </div>
    )
}

export default DetectImage;