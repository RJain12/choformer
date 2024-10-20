import React, { useState, useRef } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { ChevronRight, Download, Upload, X, Menu } from "lucide-react";
// import CHOFormerlogo_notagline from './assets/CHOFormer_logo_light.png'
import CHOFormerlogo_notagline from './assets/CHOFormer_logo_light_notagline.png'
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

const CHOExp = () => {
  // const [loading, setLoading] = useState(false);
  // const [output, setOutput] = useState(null);
  const [progress, setProgress] = useState(0);
  const [file, setFile] = useState(null);
  const [sequence, setSequence] = useState("");
  // const [error, setError] = useState(null);
  // const fileInputRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState(null);
  const [input, setInput] = useState("");
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const [isNavOpen, setIsNavOpen] = useState(false);
  const isMobile = useMediaQuery({ query: '(max-width: 768px)' });
  const navVariants = {
    open: { x: 0 },
    closed: { x: '100%' },
  };

  const toggleNav = () => setIsNavOpen(!isNavOpen);


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
        response = await axios.post(
          "https://choformer-proxy.vercel.app/proxy",
          {
            target_url: "http://3.17.139.31:8000/choexp_inference",
            sequences: [sequence],
            request_type: "POST",
          }
        );
      } else if (file) {
        const formData = new FormData();
        formData.append("file", file);
        response = await axios.post(
          "http://3.17.139.31:8000/choexp_inference",
          formData,
          {
            headers: { "Content-Type": "multipart/form-data" },
          }
        );
      } else {
        throw new Error("Please enter a sequence or choose a file.");
      }

      clearInterval(progressInterval);
      setProgress(100);
      await new Promise((resolve) => setTimeout(resolve, 500));

      setOutput(response.data.levels[0]);
    } catch (error) {
      clearInterval(progressInterval);
      setProgress(100);
      setError(
        error.response?.data?.detail || error.message || "An error occurred."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onload = (event) => {
        setSequence(event.target.result);
      };
      reader.onerror = (error) => {
        setError("Error reading file: " + error.message);
      };
      reader.readAsText(selectedFile);
    } else {
      setFile(null);
      setSequence("");
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleDownload = (format) => {
    if (output !== null) {
      const content =
        format === "csv"
          ? `Expression Level\n${output}`
          : `Expression Level: ${output}`;
      const blob = new Blob([content], {
        type: format === "csv" ? "text/csv" : "text/plain",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `cho_expression.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
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
      transition: "color 0.3s",
      fontSize: "1.1rem",
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
    },
    card: {
      background: "rgba(28, 94, 225, 0.1)",
      borderRadius: "8px",
      padding: "1.5rem",
      marginBottom: "1.5rem",
      backdropFilter: "blur(10px)",
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    },
    textarea: {
      width: "100%",
      padding: "0.5rem",
      marginBottom: "1rem",
      border: "1px solid #1c5ee1",
      borderRadius: "4px",
      fontSize: "1rem",
      color: "#fff",
      backgroundColor: "rgba(0, 0, 0, 0.2)",
      minHeight: "150px",
      resize: "vertical",
    },
    fileInputContainer: {
      display: "flex",
      alignItems: "center",
      marginBottom: "1rem",
    },
    fileInput: {
      display: "none",
    },
    // fileInputLabel: {
    //   padding: "0.5rem 1rem",
    //   backgroundColor: "#1c5ee1",
    //   color: "#fff",
    //   borderRadius: "4px",
    //   cursor: "pointer",
    //   marginRight: "1rem",
    //   transition: "background-color 0.3s",
    // },
    fileName: {
      fontSize: "0.9rem",
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
      marginBottom: "1rem",
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

    fileInputLabel: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
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
      "&:active": {
        transform: "translateY(1px)",
        boxShadow: "0 2px 4px rgba(28, 94, 225, 0.2)",
      },
    },

    expressionContainer: {
      position: "relative",
      width: "100%",
      height: "200px",
      backgroundColor: "rgba(28, 94, 225, 0.1)",
      borderRadius: "10px",
      overflow: "hidden",
      marginTop: "1rem",
    },

    expressionBubble: {
      position: "absolute",
      bottom: "10px",
      left: "50%",
      transform: "translateX(-50%)",
      width: "50px",
      height: "50px",
      backgroundColor: "#1c5ee1",
      borderRadius: "50%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      color: "#fff",
      fontWeight: "bold",
      boxShadow: "0 0 10px rgba(28, 94, 225, 0.5)",
    },

    expressionScale: {
      position: "absolute",
      bottom: "0",
      left: "0",
      width: "100%",
      height: "10px",
      backgroundColor: "rgba(255, 255, 255, 0.2)",
    },

    expressionMarker: {
      position: "absolute",
      bottom: "10px",
      width: "2px",
      height: "20px",
      backgroundColor: "#fff",
    },
    loadingBar: {
      width: "100%",
      height: "4px",
      backgroundColor: "#ccc",
      marginBottom: "1rem",
      borderRadius: "2px",
      overflow: "hidden",
    },
    loadingProgress: {
      height: "100%",
      backgroundColor: "#1c5ee1",
      transition: "width 0.5s ease-in-out",
    },
    error: {
      backgroundColor: "rgba(255, 0, 0, 0.1)",
      color: "#ff6b6b",
      padding: "1rem",
      borderRadius: "4px",
      marginBottom: "1rem",
    },
    output: {
      backgroundColor: "rgba(0, 0, 0, 0.2)",
      padding: "1rem",
      borderRadius: "4px",
      marginBottom: "1rem",
    },
    expressionMeter: {
      width: "100%",
      height: "20px",
      backgroundColor: "rgba(255, 255, 255, 0.2)",
      borderRadius: "10px",
      overflow: "hidden",
      marginTop: "1rem",
    },
    expressionLevel: {
      height: "100%",
      backgroundColor: "#1c5ee1",
      transition: "width 1s ease-in-out",
    },
    footer: {
      marginTop: "auto",
      textAlign: "center",
      padding: "1rem 0",
    },
    "@keyframes gradient": {
      "0%": { backgroundPosition: "0% 50%" },
      "50%": { backgroundPosition: "100% 50%" },
      "100%": { backgroundPosition: "0% 50%" },
    },
  };

  const ExpressionDisplay = ({ value }) => {
    const normalizedValue = Math.min(Math.max(value, 0), 1);
    const bubblePosition = `${normalizedValue * 100}%`;

    return (
      <div style={styles.expressionContainer}>
        <motion.div
          style={{
            ...styles.expressionBubble,
            left: bubblePosition,
          }}
          initial={{ y: 200 }}
          animate={{ y: 0 }}
          transition={{ type: "spring", stiffness: 120, damping: 10 }}
        >
          {(normalizedValue * 100).toFixed(1)}%
        </motion.div>
        <div style={styles.expressionScale}>
          {[0, 0.25, 0.5, 0.75, 1].map((mark) => (
            <div
              key={mark}
              style={{
                ...styles.expressionMarker,
                left: `${mark * 100}%`,
              }}
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <div style={styles.app}>
      <Header>
        <a href="/">
        <img src={CHOFormerlogo_notagline} alt="CHOFormer Logo" style={{ height: '90px', marginRight: '30px' }} />
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
              {/* <NavItem><a href="/">Home</a></NavItem> */}
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
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
          />

          <div style={styles.fileInputContainer}>
            <input
              type="file"
              accept=".csv,.fasta,.tsx,.txt"
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
            <span style={styles.fileName}>
              {file ? file.name : "No file chosen"}
            </span>
          </div>

          <motion.button
            onClick={handleGoClick}
            style={styles.button}
            disabled={loading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {loading ? "Processing..." : "Predict"}{" "}
            <ChevronRight size={18} style={{ marginLeft: "5px" }} />
          </motion.button>

          {loading && (
            <div style={styles.loadingBar}>
              <motion.div
                style={styles.loadingProgress}
                initial={{ width: "0%" }}
                animate={{ width: `${progress}%` }}
              />
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

        {output !== null && (
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
              Prediction Result
            </h2>
            <p>Expression Level: {output.toFixed(4)}</p>
            <ExpressionDisplay value={output} />
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginTop: "1rem",
              }}
            >
              {["csv", "txt"].map((format) => (
                <motion.button
                  key={format}
                  style={{ ...styles.button, width: "48%" }}
                  onClick={() => handleDownload(format)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Download {format.toUpperCase()}{" "}
                  <Download size={18} style={{ marginLeft: "5px" }} />
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
            <li>Once complete, you can view the predicted expression level and download the results in CSV or TXT format.</li>
          </ol>
        </motion.div>
      </main>

      <footer style={styles.footer}>
        <p>&copy; 2024 Choformer. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default CHOExp;