import React, { useState } from 'react';
import axios from 'axios';

const CHOExp = () => {
    const [loading, setLoading] = useState(false);
    const [output, setOutput] = useState(null);
    const [progress, setProgress] = useState(0);
    const [file, setFile] = useState(null);
    const [sequence, setSequence] = useState('');
    const [error, setError] = useState(null);

    const handleGoClick = async () => {
        setLoading(true);
        setProgress(0);
        setOutput(null);
        setError(null);

        const progressInterval = setInterval(() => {
            setProgress((prevProgress) => Math.min(prevProgress + 10, 90));
        }, 500);

        try {
            let response;
            if (sequence) {
                response = await axios.post('http://3.17.139.31:8000/choexp_inference', { sequences: [sequence] });
            } else if (file) {
                const formData = new FormData();
                formData.append('file', file);
                response = await axios.post('http://3.17.139.31:8000/choexp_inference', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
            } else {
                throw new Error("Please enter a sequence or choose a file.");
            }

            clearInterval(progressInterval);
            setProgress(100);
            await new Promise(resolve => setTimeout(resolve, 500));

            setOutput(response.data.levels[0]);
        } catch (error) {
            clearInterval(progressInterval);
            setProgress(100);
            setError(error.response?.data?.detail || error.message || "An error occurred.");
        } finally {
            setLoading(false);
        }
    };

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);
    };

    const handleDownload = (format) => {
        if (output !== null) {
            const content = format === 'csv' 
                ? `Expression Level\n${output}`
                : `Expression Level: ${output}`;
            const blob = new Blob([content], { type: format === 'csv' ? 'text/csv' : 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cho_expression.${format}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    };

    const styles = {
        app: {
            background: 'linear-gradient(270deg, #1c5ee1, hsl(0, 0%, 0%))',
            backgroundSize: '400% 400%',
            animation: 'gradient 15s ease infinite',
            color: '#fff',
            minHeight: '100vh',
            padding: '2rem',
            display: 'flex',
            flexDirection: 'column',
            fontFamily: 'Arial, sans-serif',
        },
        header: {
            marginBottom: '2rem',
        },
        nav: {
            display: 'flex',
            justifyContent: 'flex-start',
        },
        navItem: {
            marginRight: '1.5rem',
            color: '#fff',
            textDecoration: 'none',
            transition: 'color 0.3s',
            fontSize: '1.1rem',
        },
        content: {
            maxWidth: '800px',
            margin: '0 auto',
            width: '100%',
        },
        title: {
            fontSize: '2.5rem',
            marginBottom: '1.5rem',
            fontWeight: '700',
        },
        card: {
            background: 'rgba(28, 94, 225, 0.1)',
            borderRadius: '8px',
            padding: '1.5rem',
            marginBottom: '1.5rem',
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        },
        textarea: {
            width: '100%',
            padding: '0.5rem',
            marginBottom: '1rem',
            border: '1px solid #1c5ee1',
            borderRadius: '4px',
            fontSize: '1rem',
            color: '#fff',
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            minHeight: '150px',
            resize: 'vertical',
        },
        fileInputContainer: {
            display: 'flex',
            alignItems: 'center',
            marginBottom: '1rem',
        },
        fileInput: {
            display: 'none',
        },
        fileInputLabel: {
            padding: '0.5rem 1rem',
            backgroundColor: '#1c5ee1',
            color: '#fff',
            borderRadius: '4px',
            cursor: 'pointer',
            marginRight: '1rem',
            transition: 'background-color 0.3s',
        },
        fileName: {
            fontSize: '0.9rem',
        },
        button: {
            width: '100%',
            padding: '0.75rem',
            backgroundColor: '#1c5ee1',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            fontSize: '1rem',
            cursor: 'pointer',
            transition: 'background-color 0.3s',
            marginBottom: '1rem',
        },
        loadingBar: {
            width: '100%',
            height: '4px',
            backgroundColor: '#ccc',
            marginBottom: '1rem',
            borderRadius: '2px',
            overflow: 'hidden',
        },
        loadingProgress: {
            height: '100%',
            backgroundColor: '#1c5ee1',
            transition: 'width 0.5s ease-in-out',
        },
        error: {
            backgroundColor: 'rgba(255, 0, 0, 0.1)',
            color: '#ff6b6b',
            padding: '1rem',
            borderRadius: '4px',
            marginBottom: '1rem',
        },
        output: {
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            padding: '1rem',
            borderRadius: '4px',
            marginBottom: '1rem',
        },
        expressionMeter: {
            width: '100%',
            height: '20px',
            backgroundColor: 'rgba(255, 255, 255, 0.2)',
            borderRadius: '10px',
            overflow: 'hidden',
            marginTop: '1rem',
        },
        expressionLevel: {
            height: '100%',
            backgroundColor: '#1c5ee1',
            transition: 'width 1s ease-in-out',
        },
        footer: {
            marginTop: 'auto',
            textAlign: 'center',
            padding: '1rem 0',
        },
        '@keyframes gradient': {
            '0%': { backgroundPosition: '0% 50%' },
            '50%': { backgroundPosition: '100% 50%' },
            '100%': { backgroundPosition: '0% 50%' },
        },
    };

    return (
        <div style={styles.app}>
            <header style={styles.header}>
                <nav style={styles.nav}>
                    <a href="/" style={styles.navItem}>Home</a>
                    <a href="/CHOFormer" style={styles.navItem}>CHOFormer</a>
                    <a href="/CHOExp" style={styles.navItem}>CHOExp</a>
                    <a href="/about" style={styles.navItem}>About</a>
                </nav>
            </header>

            <main style={styles.content}>
                <h1 style={styles.title}>CHO Expression Predictor</h1>
                
                <div style={styles.card}>
                    <textarea
                        style={styles.textarea}
                        placeholder="Enter sequence (e.g., FASTA format)"
                        value={sequence}
                        onChange={(e) => setSequence(e.target.value)}
                    />
                    
                    <div style={styles.fileInputContainer}>
                        <input 
                            type="file" 
                            accept=".csv,.fasta,.tsx,.txt" 
                            id="fileInput" 
                            onChange={handleFileChange}
                            style={styles.fileInput}
                        />
                        <label htmlFor="fileInput" style={styles.fileInputLabel}>
                            Choose File
                        </label>
                        <span style={styles.fileName}>{file ? file.name : "No file chosen"}</span>
                    </div>
                    
                    <button 
                        onClick={handleGoClick} 
                        style={styles.button} 
                        disabled={loading}
                    >
                        {loading ? 'Processing...' : 'Predict'}
                    </button>

                    {loading && (
                        <div style={styles.loadingBar}>
                            <div style={{...styles.loadingProgress, width: `${progress}%`}}></div>
                        </div>
                    )}
                </div>

                {error && (
                    <div style={styles.error}>
                        {error}
                    </div>
                )}

                {output !== null && (
                    <div style={styles.card}>
                        <h2 style={{...styles.title, fontSize: '1.5rem', marginBottom: '1rem'}}>Prediction Result</h2>
                        <p>Expression Level: {output.toFixed(4)}</p>
                        <div style={styles.expressionMeter}>
                            <div style={{...styles.expressionLevel, width: `${output * 100}%`}}></div>
                        </div>
                        <div style={{display: 'flex', justifyContent: 'space-between', marginTop: '1rem'}}>
                            <button 
                                style={{...styles.button, width: 'auto'}}
                                onClick={() => handleDownload('csv')}
                            >
                                Download CSV
                            </button>
                            <button 
                                style={{...styles.button, width: 'auto'}}
                                onClick={() => handleDownload('txt')}
                            >
                                Download TXT
                            </button>
                        </div>
                    </div>
                )}

                <div style={styles.card}>
                    <h2 style={{...styles.title, fontSize: '1.5rem', marginBottom: '1rem'}}>How to Use This Tool</h2>
                    <ol style={{paddingLeft: '1.5rem'}}>
                        <li>Enter your sequence data in FASTA format in the text area or upload a file.</li>
                        <li>Click the "Predict" button to start the analysis.</li>
                        <li>Wait for the results to appear. This may take a few moments.</li>
                        <li>Once complete, you can view the predicted expression level and download the results in CSV or TXT format.</li>
                    </ol>
                </div>
            </main>

            <footer style={styles.footer}>
                <p>&copy; 2024 Dhruv Ramu. All rights reserved.</p>
            </footer>
        </div>
    );
};

export default CHOExp;