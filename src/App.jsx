import {useState} from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import TrainingForm from "./TrainingForm.jsx";
import Option from "./Option.jsx";
import {Route, BrowserRouter as Router, Routes} from "react-router-dom";
import DetectImage from "./DetectImage.jsx";
import DetectVideo from "./DetectVideo.jsx";

function App() {

    return (
        <Router>
            <Routes>
                <Route path="/" element={<Option/>}/>
                <Route path="/train" element={<TrainingForm/>}/>
                <Route path="/detect-image" element={<DetectImage/>}/>
                <Route path="/detect-video" element={<DetectVideo/>}/>
            </Routes>
        </Router>)
}

export default App
