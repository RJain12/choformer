import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Home';
import About from './About';
import CHOFormer from './CHOFormer';
import CHOExp from './CHOExp';
const App = () => {
    return (
        <Router>
            <Routes>
                <Route exact path="/" element={<Home />} />
                <Route path="/about" element={<About />} />
                <Route path="/CHOFormer" element={<CHOFormer />} />
                <Route path = "/CHOExp" element={<CHOExp/>}/>
            </Routes>
        </Router>
    );
};

export default App;