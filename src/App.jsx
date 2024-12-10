import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import TrainingForm from "./TrainingForm.jsx";

function App() {
  const [count, setCount] = useState(0)

  return (
    <TrainingForm></TrainingForm>
  )
}

export default App
