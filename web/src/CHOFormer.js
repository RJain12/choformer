import React, { useState, useRef } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { ChevronRight, Download, Upload } from "lucide-react";

const CHOFormer = () => {
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState(null);
  const [input, setInput] = useState("");
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const handleGoClick = async () => {
    setLoading(true);
    setError("");
    setOutput(null);

    try {
      const response = await axios.post(
        "https://choformer-proxy.vercel.app/proxy",
        {
          target_url: "http://3.17.139.31:7999/choformer_inference",
          sequences: [input],
          request_type: "POST",
        }
      );
      setOutput(response.data.sequences[0]);
    } catch (err) {
      setError(
        "An error occurred while processing your request. Please try again."
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (event) => {
        setInput(event.target.result);
      };
      reader.onerror = (error) => {
        setError("Error reading file: " + error.message);
      };
      reader.readAsText(file);
    } else {
      setFileName("");
      setInput("");
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const downloadFile = (content, fileType) => {
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `output.${fileType}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const styles = {
    uploadButton: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "0.5rem 1rem",
        backgroundColor: "#1c5ee1",
        color: "#fff",
        border: "none",
        borderRadius: "50px",
        fontSize: "0.9rem",
        fontWeight: "600",
        cursor: "pointer",
        transition: "all 0.3s ease",
        marginRight: "1rem",
        boxShadow: "0 4px 6px rgba(28, 94, 225, 0.2)",
        "&:hover": {
          backgroundColor: "#4d7ce9",
          transform: "translateY(-2px)",
          boxShadow: "0 6px 8px rgba(28, 94, 225, 0.3)",
        },
      },
  
    app: {
      background: "linear-gradient(270deg, #1c5ee1, hsl(0, 0%, 0%))",
      backgroundSize: "400% 400%",
      animation: "gradient 15s ease infinite",
      color: "#fff",
      minHeight: "100vh",
      padding: "2rem",
      display: "flex",
      flexDirection: "column",
      fontFamily: "Arial, sans-serif",
    },
    header: {
      marginBottom: "2rem",
    },
    nav: {
      display: "flex",
      justifyContent: "flex-start",
    },
    navItem: {
      marginRight: "1.5rem",
      color: "#fff",
      textDecoration: "none",
      transition: "all 0.3s ease",
      fontSize: "1.1rem",
      position: "relative",
      "&::after": {
        content: '""',
        position: "absolute",
        width: "0%",
        height: "2px",
        bottom: "-4px",
        left: "50%",
        backgroundColor: "#1c5ee1",
        transition: "width 0.3s ease, left 0.3s ease",
      },
      "&:hover::after": {
        width: "100%",
        left: "0%",
      },
    },
    content: {
      maxWidth: "800px",
      margin: "0 auto",
      width: "100%",
    },
    title: {
      fontSize: "2.5rem",
      marginBottom: "1.5rem",
      fontWeight: "700",
      textShadow: "2px 2px 4px rgba(0, 0, 0, 0.3)",
    },
    card: {
      background: "rgba(28, 94, 225, 0.1)",
      borderRadius: "12px",
      padding: "1.5rem",
      marginBottom: "1.5rem",
      backdropFilter: "blur(10px)",
      boxShadow: "0 8px 32px rgba(0, 0, 0, 0.1)",
      transition: "transform 0.3s ease, box-shadow 0.3s ease",
      "&:hover": {
        transform: "translateY(-5px)",
        boxShadow: "0 12px 48px rgba(0, 0, 0, 0.15)",
      },
    },
    textarea: {
      width: "100%",
      padding: "0.75rem",
      marginBottom: "1rem",
      border: "1px solid rgba(28, 94, 225, 0.5)",
      borderRadius: "8px",
      fontSize: "1rem",
      color: "#fff",
      backgroundColor: "rgba(0, 0, 0, 0.2)",
      minHeight: "150px",
      resize: "vertical",
      transition: "border-color 0.3s ease, box-shadow 0.3s ease",
      "&:focus": {
        outline: "none",
        borderColor: "#1c5ee1",
        boxShadow: "0 0 0 3px rgba(28, 94, 225, 0.3)",
      },
    },
    fileInputContainer: {
      display: "flex",
      alignItems: "center",
      marginBottom: "1rem",
    },
    fileInput: {
      display: "none",
    },
    fileInputLabel: {
      display: "flex",
      alignItems: "center",
      padding: "0.5rem 1rem",
      backgroundColor: "#1c5ee1",
      color: "#fff",
      borderRadius: "50px",
      cursor: "pointer",
      marginRight: "1rem",
      transition: "all 0.3s ease",
      fontWeight: "600",
      boxShadow: "0 4px 6px rgba(28, 94, 225, 0.2)",
      "&:hover": {
        backgroundColor: "#4d7ce9",
        transform: "translateY(-2px)",
        boxShadow: "0 6px 8px rgba(28, 94, 225, 0.3)",
      },
    },
    fileName: {
      fontSize: "0.9rem",
      opacity: 0.8,
    },
    button: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      width: "100%",
      padding: "0.75rem",
      backgroundColor: "#1c5ee1",
      color: "#fff",
      border: "none",
      borderRadius: "50px",
      fontSize: "1rem",
      fontWeight: "600",
      cursor: "pointer",
      transition: "all 0.3s ease",
      boxShadow: "0 4px 6px rgba(28, 94, 225, 0.2)",
      "&:hover": {
        backgroundColor: "#4d7ce9",
        transform: "translateY(-2px)",
        boxShadow: "0 6px 8px rgba(28, 94, 225, 0.3)",
      },
      "&:active": {
        transform: "translateY(1px)",
        boxShadow: "0 2px 4px rgba(28, 94, 225, 0.2)",
      },
    },
    loadingBar: {
      width: "100%",
      height: "4px",
      backgroundColor: "rgba(255, 255, 255, 0.2)",
      marginTop: "1rem",
      borderRadius: "2px",
      overflow: "hidden",
    },
    loadingProgress: {
      width: "30%",
      height: "100%",
      backgroundColor: "#1c5ee1",
      borderRadius: "2px",
      animation: "loading 1.5s infinite",
    },
    error: {
      backgroundColor: "rgba(255, 107, 107, 0.1)",
      color: "#ff6b6b",
      padding: "1rem",
      borderRadius: "8px",
      marginBottom: "1rem",
      boxShadow: "0 4px 6px rgba(255, 107, 107, 0.1)",
    },
    output: {
        backgroundColor: "rgba(0, 0, 0, 0.2)",
        padding: "1rem",
        borderRadius: "8px",
        fontSize: "0.9rem",
        lineHeight: "1.5",
        wordWrap: "break-word",
        whiteSpace: "pre-wrap",
        maxWidth: "100%",
      },
    footer: {
      marginTop: "auto",
      textAlign: "center",
      padding: "1rem 0",
      color: "rgba(255, 255, 255, 0.7)",
    },
    downloadButton: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "0.5rem 1rem",
      backgroundColor: "#1c5ee1",
      color: "#fff",
      border: "none",
      borderRadius: "50px",
      fontSize: "0.9rem",
      fontWeight: "600",
      cursor: "pointer",
      transition: "all 0.3s ease",
      marginRight: "1rem",
      boxShadow: "0 4px 6px rgba(28, 94, 225, 0.2)",
      "&:hover": {
        backgroundColor: "#4d7ce9",
        transform: "translateY(-2px)",
        boxShadow: "0 6px 8px rgba(28, 94, 225, 0.3)",
      },
    },
    "@keyframes loading": {
      "0%": { transform: "translateX(-100%)" },
      "100%": { transform: "translateX(400%)" },
    },
    "@keyframes gradient": {
      "0%": { backgroundPosition: "0% 50%" },
      "50%": { backgroundPosition: "100% 50%" },
      "100%": { backgroundPosition: "0% 50%" },
    },
  };

  const formatOutput = (output) => {
    // Split the output into lines
    console.log('o', output)
    const lines = output.split('\n');
    
    // Process each line
    const formattedLines = lines.map(line => {
      // If the line is a FASTA header (starts with '>'), don't wrap it
      if (line.startsWith('>')) {
        return line;
      }
      // For sequence lines, wrap every 60 characters
      return line.match(/.{1,60}/g).join('\n');
    });

    // Join the lines back together
    return formattedLines.join('\n');
  };

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <nav style={styles.nav}>
          {["Home", "CHOFormer", "CHOExp", "About"].map((item) => (
            <motion.a
              key={item}
              href={`/${item.toLowerCase()}`}
              style={styles.navItem}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {item}
            </motion.a>
          ))}
        </nav>
      </header>

      <main style={styles.content}>
        <motion.h1
          style={styles.title}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          CHO Expression Predictor
        </motion.h1>

        <motion.div
          style={styles.card}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <textarea
            style={styles.textarea}
            placeholder="Enter sequence (e.g., FASTA format) or upload a file"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />

          <div style={styles.fileInputContainer}>
            <input
              type="file"
              accept=".csv,.fasta,.exe,.txt"
              id="fileInput"
              ref={fileInputRef}
              onChange={handleFileChange}
              style={styles.fileInput}
            />
            <motion.button
              onClick={handleUploadClick}
              style={styles.uploadButton}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Upload File <Upload size={18} style={{ marginLeft: "5px" }} />
            </motion.button>
            <span style={styles.fileName}>{fileName || "No file chosen"}</span>
          </div>

          <motion.button
            onClick={handleGoClick}
            style={styles.button}
            disabled={loading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {loading ? "Processing..." : "Predict"} <ChevronRight size={18} style={{ marginLeft: "5px" }} />
          </motion.button>

          {loading && (
            <div style={styles.loadingBar}>
              <div style={styles.loadingProgress}></div>
            </div>
          )}
        </motion.div>

        {error && (
          <motion.div
            style={styles.error}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {error}
          </motion.div>
        )}

        {output && (
          <motion.div
            style={styles.card}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h2
              style={{
                ...styles.title,
                fontSize: "1.5rem",
                marginBottom: "1rem",
              }}
            >
              Prediction Results
            </h2>
            <pre style={styles.output}>{formatOutput(output)}</pre>
            <div
              style={{
                display: "flex",
                justifyContent: "flex-start",
                marginTop: "1rem",
              }}
            >
              {["fasta", "txt"].map((format) => (
                <motion.button
                  key={format}
                  style={styles.downloadButton}
                  onClick={() => downloadFile(output, format)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Download {format.toUpperCase()} <Download size={18} style={{ marginLeft: "5px" }} />
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}

        <motion.div
          style={styles.card}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <h2
            style={{
              ...styles.title,
              fontSize: "1.5rem",
              marginBottom: "1rem",
            }}
          >
            How to Use This Tool
          </h2>
          <ol style={{ paddingLeft: "1.5rem" }}>
            <li>Enter your sequence data in FASTA format in the text area or upload a file.</li>
            <li>If you upload a file, its contents will be displayed in the text area for review or editing.</li>
            <li>Click the "Predict" button to start the analysis.</li>
            <li>Wait for the results to appear. This may take a few moments.</li>
            <li>Once complete, you can view the results and download them in FASTA or TXT format.</li>
          </ol>
        </motion.div>
      </main>

      <footer style={styles.footer}>
        <p>&copy; 2024 Dhruv Ramu. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default CHOFormer;