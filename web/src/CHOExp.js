import React, { useState, } from 'react';
import styles from './CHOExp.module.css';
import { API_URL } from './constants';

const CHOExp = () => {
    const [loading, setLoading] = useState(false);
    const [output, setOutput] = useState(null);
    const [progress, setProgress] = useState(0); // Progress state for the loading bar
    const [file, setFile] = useState(''); // State to store the chosen file name
    const [sequence, setSequence] = useState('');
    const [error, setError] = useState(null);

    const handleGoClick = async () => {
        setLoading(true);
        setProgress(0); // Reset progress
        setOutput(null); // Reset output
        setError(null); // Reset error

        if (sequence) {
            const progressInterval = setInterval(() => {
                setProgress((prevProgress) => {
                    if (prevProgress >= 90) {
                        return prevProgress;
                    }
                    return prevProgress + 10;
                });
            }, 500);

            try {
                const response = await fetch(`${API_URL}/process`, {
                    method: 'POST',
                    headers: {
                        'Content-type': 'application/json'
                    },
                    body: JSON.stringify({ text: sequence })
                });

                clearInterval(progressInterval);
                setProgress(100);

                await new Promise(resolve => setTimeout(resolve, 500)); // Delay to show 100% progress

                const data = await response.json();
                if (response.ok) {
                    setOutput(data.result);
                } else {
                    setError(data.detail);
                }
            } catch (error) {
                clearInterval(progressInterval);
                setProgress(100);
                setError("An error occurred while processing the request.");
            } finally {
                setLoading(false);
            }
        }
        else if (file) {
            const progressInterval = setInterval(() => {
                setProgress((prevProgress) => {
                    if (prevProgress >= 90) {
                        return prevProgress;
                    }
                    return prevProgress + 10;
                });
            }, 500);

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch(`${API_URL}/process_csv`, {
                    method: 'POST',
                    body: formData
                });

                clearInterval(progressInterval);
                setProgress(100);

                await new Promise(resolve => setTimeout(resolve, 500)); // Delay to show 100% progress

                const data = await response.json();
                if (response.ok) {
                    setOutput(data.results);
                } else {
                    setError(data.detail);
                }
            } catch (error) {
                clearInterval(progressInterval);
                setProgress(100);
                setError("An error occurred while processing the file.");
            } finally {
                setLoading(false);
            }
        }

        else {

            setError("Please enter a sequence or choose a file.");
            setOutput(null);
            setLoading(false);

        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        setFile(file);
    };

    function handleDownloadTXT() {
        if (output) {
            const blob = new Blob([output], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'output.txt';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }
    }


    function handleDownloadCSV() {
        if (output) {
            const blob = new Blob([
                "result\n" + (Array.isArray(output) ? output
                    .filter(item => typeof item === 'string')
                    .join('\n') : output)
            ], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'output.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }
    }

    return (
        <div className={styles.app}>
            <header className={styles.header}>
                <nav>
                    <ul className={styles.navUl}>
                        <li className={styles.navLi}><a href="/" className={styles.navA}>Home</a></li>
                        <li className={styles.navLi}><a href="/CHOFormer" className={styles.navA}>CHOFormer</a></li>
                        <li className={styles.navLi}><a href="/CHOExp" className={styles.navA}>CHOExp</a></li>
                        <li className={styles.navLi}><a href="/about" className={styles.navA}>About</a></li>

                    </ul>
                </nav>
            </header>
            <div className={styles.contentContainer}>
                <main>
                    <section className={styles.content}>
                        <div className={styles.predictorContainer}>
                            <h1 className={styles.contentH1}>CHO Expression Predictor</h1>
                            <textarea
                                placeholder="Enter sequence (e.g., FASTA format)"
                                className={styles.textarea}
                                value={sequence}
                                onChange={(e) => setSequence(e.target.value)}
                            />
                        </div>
                        {/* Hidden File Input */}
                        <input
                            type="file"
                            accept=".csv,.fasta,.tsx,.txt"
                            className={styles.hiddenInput}
                            id="fileInput"
                            onChange={handleFileChange}
                        />
                        {/* Custom File Button */}
                        <label htmlFor="fileInput" className={styles.customFileButton}>
                            <span>Choose File</span>
                            <span className={styles.fileName}>{file?.name || "No file chosen"}</span>
                        </label>
                        <button
                            className={styles.button}
                            onClick={handleGoClick}
                            disabled={loading}
                        >
                            Go!
                        </button>
                        {
                            loading &&
                            <div className={styles.loadingBar}>
                                <div style={{
                                    height: '100%',
                                    backgroundColor: '#1c5ee1',
                                    transition: 'width 0.1s ease-in-out',
                                    width: `${progress}%`
                                }}

                                />
                            </div>
                        }


                        <div className={styles.outputContainer}>
                            <div className={styles.output}>
                                {
                                    error ? <p

                                        style={{
                                            color: 'red'
                                        }}
                                    >
                                        {error}

                                    </p> :
                                        <>
                                            {
                                                Array.isArray(output) ? output.map((item, index) => (
                                                    <p key={index}>{item}</p>
                                                )) : <p>{output}</p>
                                            }
                                        </>
                                }
                                {
                                    output != null &&
                                    <>
                                        <button
                                            className={styles.button}
                                            onClick={handleDownloadCSV}
                                        >
                                            Download as CSV
                                        </button>
                                        <button
                                            className={styles.button}
                                            onClick={handleDownloadTXT}
                                        >
                                            Download as TXT
                                        </button>
                                    </>
                                }
                            </div>
                        </div>

                        <h2 className={styles.contentH1}>How to Use This Tool</h2>
                        <p className={styles.contentP}>1. Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                        <p className={styles.contentP}>2. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
                        <p className={styles.contentP}>3. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.</p>
                    </section>
                </main>
            </div>
            <footer className={styles.footer}>
                <p className={styles.footerP}>&copy; 2024 Dhruv Ramu. All rights reserved.</p>
            </footer>
        </div >
    );
};

export default CHOExp;