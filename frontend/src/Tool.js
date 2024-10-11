import React, { useState, useEffect } from 'react';

const Tool = () => {
    const [loading, setLoading] = useState(false);
    const [output, setOutput] = useState(null);
    const [progress, setProgress] = useState(0); // Progress state for the loading bar
    const [fileName, setFileName] = useState(''); // State to store the chosen file name

    const handleGoClick = () => {
        setLoading(true);
        setProgress(0); // Reset progress
        setOutput(null); // Reset output

        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev < 100) {
                    return prev + 5; // Increment progress
                }
                clearInterval(interval);
                return prev;
            });
        }, 100); // Update every 100ms

        // Simulate loading completion after 2 seconds
        setTimeout(() => {
            clearInterval(interval);
            setLoading(false);
            setOutput("Sample output data...");
        }, 2000);
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        setFileName(file ? file.name : '');
    };

    const styles = {
        app: {
            background: 'linear-gradient(270deg, #1c5ee1, hsl(0, 0%, 0%))',
            backgroundSize: '400% 400%',
            animation: 'gradient 15s ease infinite',
            color: '#fff',
            minHeight: '100vh',
            padding: '0 2rem',
            display: 'flex',
            flexDirection: 'column',
        },
        header: {
            background: 'transparent',
            color: '#fff',
            padding: '1rem 0',
            width: '100%',
            textAlign: 'left',
        },
        navUl: {
            listStyle: 'none',
            display: 'flex',
            justifyContent: 'flex-start',
            padding: 0,
        },
        navLi: {
            margin: '0 1rem',
        },
        navA: {
            color: '#fff',
            textDecoration: 'none',
        },
        contentContainer: {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            justifyContent: 'center',
            width: '100%',
            height: '100%',
        },
        predictorContainer: {
            display: 'flex',
            flexDirection: 'column', // Stacks title and input field vertically
            alignItems: 'flex-start',
            justifyContent: 'flex-start',
            width: '100%',
            maxWidth: '600px',
        },
        content: {
            textAlign: 'left',
            padding: '2rem 0',
            background: 'transparent',
            width: '100%',
            maxWidth: '600px',
        },
        contentH1: {
            margin: '0 0 0.5rem 0', // Adds space below the heading
            fontSize: '2.5rem',
            textAlign: 'left',
        },
        contentP: {
            fontSize: '1.5rem',
            margin: '0 0 2rem 0',
            textAlign: 'left',
        },
        textarea: {
            width: '100%',
            padding: '0.5rem',
            margin: '0.5rem 0',
            border: '1px solid #1c5ee1',
            borderRadius: '4px',
            fontSize: '1rem',
            color: '#fff',
            backgroundColor: 'transparent',
            boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)',
            transition: 'border-color 0.3s',
            minHeight: '150px', // Makes the textarea bigger
        },
        textareaFocus: {
            borderColor: '#fff',
            outline: 'none',
        },
        loadingBar: {
            width: '100%',
            height: '1rem',
            backgroundColor: '#ccc',
            margin: '1rem 0',
            position: 'relative',
            overflow: 'hidden',
        },
        loadingProgress: {
            width: `${progress}%`,
            height: '100%',
            backgroundColor: '#1c5ee1',
            transition: 'width 0.1s ease-in-out',
        },
        outputContainer: {
            display: 'flex',
            width: '100%',
            marginTop: '2rem',
        },
        output: {
            margin: '1rem 0',
            padding: '1rem',
            backgroundColor: 'transparent',
            color: '#fff',
            width: '100%',
            maxWidth: '800px',
            textAlign: 'left',
            borderRadius: '8px',
        },
        footer: {
            textAlign: 'left',
            padding: '1rem 0',
            background: 'transparent',
            color: '#fff',
            width: '100%',
        },
        footerP: {
            marginTop: '-25px',
        },
        button: {
            padding: '0.5rem 1rem',
            backgroundColor: '#1c5ee1',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginTop: '1rem',
            transition: 'background-color 0.3s, transform 0.3s',
            fontSize: '1rem',
            width: '100%',
            boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)',
        },
        buttonHover: {
            backgroundColor: '#fff',
            color: '#1c5ee1',
            transform: 'scale(1.05)',
        },
        // Custom styles for the file input
        hiddenInput: {
            display: 'none',
        },
        customFileButton: {
            padding: '0.5rem 1rem',
            backgroundColor: '#1c5ee1',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginTop: '1rem',
            transition: 'background-color 0.3s, transform 0.3s',
            fontSize: '1rem',
            width: '100%',
            boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)',
            display: 'flex',
            justifyContent: 'space-between',
        },
        fileName: {
            fontSize: '1rem',
            color: '#fff',
        },
    };

    return (
        <div style={styles.app}>
            <header style={styles.header}>
                <nav>
                    <ul style={styles.navUl}>
                        <li style={styles.navLi}><a href="/" style={styles.navA}>Home</a></li>
                        <li style={styles.navLi}><a href="/about" style={styles.navA}>About</a></li>
                        <li style={styles.navLi}><a href="/CHOFormer" style={styles.navA}>CHOFormer</a></li>
                        <li style={styles.navLi}><a href="/CHOExp" style={styles.navA}>CHOExp</a></li>
                    </ul>
                </nav>
            </header>
            <div style={styles.contentContainer}>
                <main>
                    <section style={styles.content}>
                        <div style={styles.predictorContainer}>
                            <h1 style={styles.contentH1}>CHO Expression Predictor</h1>
                            <textarea
                                placeholder="Enter sequence (e.g., FASTA format)"
                                style={styles.textarea}
                                onFocus={(e) => e.currentTarget.style.borderColor = styles.textareaFocus.borderColor}
                                onBlur={(e) => e.currentTarget.style.borderColor = styles.textarea.border}
                            />
                        </div>
                        {/* Hidden File Input */}
                        <input 
                            type="file" 
                            accept=".csv,.fasta,.exe" 
                            style={styles.hiddenInput} 
                            id="fileInput" 
                            onChange={handleFileChange} 
                        />
                        {/* Custom File Button */}
                        <label htmlFor="fileInput" style={styles.customFileButton}>
                            <span>Choose File</span>
                            <span style={styles.fileName}>{fileName || "No file chosen"}</span>
                        </label>
                        <button 
                            style={styles.button} 
                            onClick={handleGoClick} 
                            onMouseOver={(e) => e.currentTarget.style.backgroundColor = styles.buttonHover.backgroundColor} 
                            onMouseOut={(e) => e.currentTarget.style.backgroundColor = styles.button.backgroundColor}
                        >
                            Go!
                        </button>
                        {loading && (
                            <div style={styles.loadingBar}>
                                <div style={styles.loadingProgress}></div>
                            </div>
                        )}
                        {output && (
                            <div style={styles.outputContainer}>
                                <div style={styles.output}>
                                    <p>{output}</p>
                                    <button 
                                        style={styles.button}
                                        onMouseOver={(e) => e.currentTarget.style.backgroundColor = styles.buttonHover.backgroundColor} 
                                        onMouseOut={(e) => e.currentTarget.style.backgroundColor = styles.button.backgroundColor}
                                    >
                                        Download as CSV
                                    </button>
                                    <button 
                                        style={styles.button}
                                        onMouseOver={(e) => e.currentTarget.style.backgroundColor = styles.buttonHover.backgroundColor} 
                                        onMouseOut={(e) => e.currentTarget.style.backgroundColor = styles.button.backgroundColor}
                                    >
                                        Download as TXT
                                    </button>
                                </div>
                            </div>
                        )}
                        <h2 style={styles.contentH1}>How to Use This Tool</h2>
                        <p style={styles.contentP}>1. Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                        <p style={styles.contentP}>2. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
                        <p style={styles.contentP}>3. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.</p>
                    </section>
                </main>
            </div>
            <footer style={styles.footer}>
                <p style={styles.footerP}>&copy; 2024 Dhruv Ramu. All rights reserved.</p>
            </footer>
        </div>
    );
};

export default Tool;
