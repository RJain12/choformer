import React, { useState } from 'react';
import axios from 'axios';

const CHOFormer = () => {
    const [loading, setLoading] = useState(false);
    const [output, setOutput] = useState(null);
    const [input, setInput] = useState('');
    const [fileName, setFileName] = useState('');
    const [error, setError] = useState('');

    const handleGoClick = async () => {
        setLoading(true);
        setError('');
        setOutput(null);

        try {
            const response = await axios.post('http://3.17.139.31:7999/choformer_inference', {
                sequences: [input]
            });
            setOutput(response.data.sequences);
        } catch (err) {
            setError('An error occurred while processing your request. Please try again.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        setFileName(file ? file.name : '');
        // Here you would typically read the file contents and set the input
    };

    const downloadFile = (content, fileType) => {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `output.${fileType}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
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
        },
        loadingBar: {
            width: '100%',
            height: '4px',
            backgroundColor: '#ccc',
            marginTop: '1rem',
            overflow: 'hidden',
            position: 'relative',
        },
        loadingProgress: {
            width: '30%',
            height: '100%',
            backgroundColor: '#1c5ee1',
            position: 'absolute',
            animation: 'loading 1.5s infinite',
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
            whiteSpace: 'pre-wrap',
            overflowX: 'auto',
        },
        footer: {
            marginTop: 'auto',
            textAlign: 'center',
            padding: '1rem 0',
        },
        downloadButton: {
            padding: '0.5rem 1rem',
            backgroundColor: '#1c5ee1',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            fontSize: '1rem',
            cursor: 'pointer',
            transition: 'background-color 0.3s',
            marginRight: '1rem',
        },
        '@keyframes loading': {
            '0%': { left: '-30%' },
            '100%': { left: '100%' },
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
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                    />
                    
                    <div style={styles.fileInputContainer}>
                        <input 
                            type="file" 
                            accept=".csv,.fasta,.exe" 
                            id="fileInput" 
                            onChange={handleFileChange}
                            style={styles.fileInput}
                        />
                        <label htmlFor="fileInput" style={styles.fileInputLabel}>
                            Choose File
                        </label>
                        <span style={styles.fileName}>{fileName || "No file chosen"}</span>
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
                            <div style={styles.loadingProgress}></div>
                        </div>
                    )}
                </div>

                {error && (
                    <div style={styles.error}>
                        {error}
                    </div>
                )}

                {output && (
                    <div style={styles.card}>
                        <h2 style={{...styles.title, fontSize: '1.5rem', marginBottom: '1rem'}}>Prediction Results</h2>
                        <pre style={styles.output}>
                            {output}
                        </pre>
                        <div style={{display: 'flex', justifyContent: 'flex-start', marginTop: '1rem'}}>
                            <button 
                                style={styles.downloadButton}
                                onClick={() => downloadFile(output, 'fasta')}
                            >
                                Download FASTA
                            </button>
                            <button 
                                style={styles.downloadButton}
                                onClick={() => downloadFile(output, 'txt')}
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
                        <li>Once complete, you can view the results and download them in FASTA or TXT format.</li>
                    </ol>
                </div>
            </main>

            <footer style={styles.footer}>
                <p>&copy; 2024 Dhruv Ramu. All rights reserved.</p>
            </footer>
        </div>
    );
};

export default CHOFormer;