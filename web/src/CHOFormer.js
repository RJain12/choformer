import React, { useState, useRef } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { ChevronRight, Download, Upload, X, Menu } from "lucide-react";
import { useMediaQuery } from 'react-responsive';
import styled, { keyframes, createGlobalStyle } from 'styled-components';


const GlobalStyle = createGlobalStyle`
  body {
    margin: 0;g
    padding: 0;
    box-sizing: border-box;
    font-family: 'Inter', sans-serif;
  }
`;

const AppContainer = styled.div`
  background: radial-gradient(64.1% 57.6% at 68.3% 44%, #1c5ee1 10.56%, hsl(0, 0%, 0%) 100%);
  color: #fff;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
`;

const Header = styled.header`
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Nav = styled(motion.nav)`
  @media (max-width: 768px) {
    position: fixed;
    top: 0;
    right: 0;
    height: 100vh;
    background: rgba(28, 94, 225, 0.95);
    padding: 2rem;
    z-index: 1000;
  }
`;

const NavList = styled.ul`
  list-style: none;
  display: flex;
  gap: 2rem;

  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

const NavItem = styled.li`
  a {
    color: #fff;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.3s ease;

    &:hover {
      color: #4d90fe;
    }
  }
`;

const MenuButton = styled.button`
  background: none;
  border: none;
  color: #fff;
  font-size: 1.5rem;
  cursor: pointer;
  display: none;

  @media (max-width: 768px) {
    display: block;
  }
`;

const ContentContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;

  @media (min-width: 1024px) {
    flex-direction: row;
    align-items: flex-start;
    justify-content: space-between;
  }
`;

const Content = styled.section`
  text-align: left;
  max-width: 800px;
`;

const Title = styled(motion.h1)`
  font-size: 2.5rem;
  margin-bottom: 1rem;
  color: #4d90fe;
`;

const Paragraph = styled(motion.p)`
  font-size: 1.1rem;
  line-height: 1.6;
  margin-bottom: 1.5rem;
`;

const Highlight = styled.span`
  color: #4d90fe;
  font-weight: 600;
`;

const LogoContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2rem;
  margin-top: 2rem;

  @media (min-width: 1024px) {
    margin-top: 0;
  }
`;

const Logo = styled(motion.img)`
  height: 80px;
  width: auto;
`;

const Footer = styled.footer`
  text-align: center;
  padding: 1rem;
  margin-top: auto;
`;

const CHOFormer = () => {
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState(null);
  const [input, setInput] = useState("");
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const [isNavOpen, setIsNavOpen] = useState(false);
  const isMobile = useMediaQuery({ query: '(max-width: 768px)' });

  const toggleNav = () => setIsNavOpen(!isNavOpen);

  const navVariants = {
    open: { x: 0 },
    closed: { x: '100%' },
  };

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
      <Header>
        <a href="/">
        <img src="/CHOFormer_logo.png" alt="CHOFormer Logo" style={{ height: '60px', marginRight: '20px' }} />
    </a>
          <MenuButton onClick={toggleNav}>
            {isNavOpen ? <X /> : <Menu />}
          </MenuButton>
          <Nav
            initial={false}
            animate={isMobile ? (isNavOpen ? "open" : "closed") : "open"}
            variants={navVariants}
          >
            <NavList>
              <NavItem><a href="/">Home</a></NavItem>
              <NavItem><a href="/CHOFormer">CHOFormer</a></NavItem>
              <NavItem><a href="/CHOExp">CHOExp</a></NavItem>
              <NavItem><a href="/about">About</a></NavItem>
            </NavList>
          </Nav>
        </Header>

      <main style={styles.content}>
        <motion.h1
          style={styles.title}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          CHO Codon Optimization (CHOFormer)
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
        <p>&copy; 2024 Choformer. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default CHOFormer;