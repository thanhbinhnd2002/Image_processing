import React from "react";
import { useNavigate } from "react-router-dom";

const Option = () => {
    const navigate = useNavigate(); // Hook to handle navigation

    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h2>Choose an Option</h2>
            <div style={{ display: "flex", justifyContent: "center", gap: "20px", marginTop: "20px" }}>
                <button
                    onClick={() => navigate("/train")}
                    style={{
                        padding: "10px 20px",
                        fontSize: "16px",
                        cursor: "pointer",
                        backgroundColor: "#4CAF50",
                        color: "white",
                        border: "none",
                        borderRadius: "5px",
                    }}
                >
                    Training Model
                </button>
                <button
                    onClick={() => navigate("/detect-image")}
                    style={{
                        padding: "10px 20px",
                        fontSize: "16px",
                        cursor: "pointer",
                        backgroundColor: "#2196F3",
                        color: "white",
                        border: "none",
                        borderRadius: "5px",
                    }}
                >
                    Detect in Image
                </button>
                <button
                    onClick={() => navigate("/detect-video")}
                    style={{
                        padding: "10px 20px",
                        fontSize: "16px",
                        cursor: "pointer",
                        backgroundColor: "#FF5722",
                        color: "white",
                        border: "none",
                        borderRadius: "5px",
                    }}
                >
                    Detect in Video
                </button>
            </div>
        </div>
    );
};

export default Option;
