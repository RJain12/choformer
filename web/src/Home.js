import React, { useState } from 'react';
import { useMediaQuery } from 'react-responsive';
import styled, { keyframes } from 'styled-components';
import gene from './assets/gene.png';
import ncbiLogo from './assets/ncbi.png'; // Import the NCBI logo
import openaiLogo from './assets/openai.png'; // Import the OpenAI logo
import awsLogo from './assets/aws.png'; // Import the AWS logo
import esmlogo from './assets/esm.png';
import { Link } from "react-router-dom";

// Styled Components
const AppContainer = styled.div`
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    text-align: left;
    background: radial-gradient(64.1% 57.6% at 68.3% 44%, #1c5ee1 10.56%, hsl(0, 0%, 0%) 100%);
    color: #fff;
    min-height: 100vh;
    padding: 0 2rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;

    @media (max-width: 768px) {
        padding: 0 1rem; // Adjust padding for smaller screens
    }
`;

const Header = styled.header`
    background: transparent;
    color: #fff;
    padding: 1rem 0;
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
`;

const NavToggle = styled.div`
    display: block; // Always show the toggle
    font-size: 1.5rem;
    cursor: pointer;

    @media (min-width: 769px) {
        display: none; // Hide toggle on larger screens
    }
`;

// Keyframes for animation
const slideDown = keyframes`
    from {
        max-height: 0;
        opacity: 0;
    }
    to {
        max-height: 500px; // Set a maximum height for the dropdown
        opacity: 1;
    }
`;

const slideUp = keyframes`
    from {
        max-height: 500px; // Set a maximum height for the dropdown
        opacity: 1;
    }
    to {
        max-height: 0;
        opacity: 0;
    }
`;

const NavUl = styled.ul`
    list-style: none;
    padding: 0;
    margin: 0;
    overflow: hidden; // Hide overflow for animation
    display: flex;

    @media (max-width: 768px) {
        display: block; // Use block display for mobile
        background-color: #1a1a1a; // Dark background for mobile navbar
        position: fixed; // Fixed position to start from the top
        top: 0; // Start from the top of the screen
        left: 0;
        right: 0; // Full width
        transition: opacity 0.3s ease; // Smooth transition for opacity
        border-radius: 8px; // Slightly rounded corners
        z-index: 1000; // Ensure it appears on top
        padding: 1rem 0; // Padding for mobile navbar
        animation: ${props => (props.isOpen ? slideDown : slideUp)} 0.3s ease forwards; // Animation on open/close
    }
`;

const NavLi = styled.li`
    margin: 0 1rem;

    @media (max-width: 768px) {
        margin: 1rem 0; // Space out items on mobile
        padding: 0.5rem 1rem; // Padding for each item
        text-align: center; // Center text for mobile
    }
`;

const NavA = styled.a`
    color: #fff;
    text-decoration: none;
    display: block; // Make the entire area clickable

    &:hover {
        background: rgba(255, 255, 255, 0.2); // Light background on hover
    }
`;

const CloseButton = styled.div`
    cursor: pointer;
    color: #fff;
    font-size: 1.5rem;
    text-align: right;
    padding: 1rem;

    @media (min-width: 769px) {
        display: none; // Hide on larger screens
    }
`;

const HeroContainer = styled.div`
    display: flex;
    justify-content: space-between;
    align-items: flex-start; // Align items at the start
    width: 100%;

    @media (max-width: 768px) {
        flex-direction: column; // Stack hero content on mobile
    }
`;

const Hero = styled.section`
    text-align: left;
    padding: 2rem 0;
    background: transparent;
    width: 70%;

    @media (max-width: 768px) {
        width: 100%; // Full width on mobile
    }
`;

const HeroH1 = styled.h1`
    margin: 0 0 1rem 0; // Increased bottom margin for spacing
    font-size: 4rem;
`;

const HeroP = styled.p`
    font-size: 1.5rem;
    margin: 0 0 2rem 0;
`;

const HeroButton = styled(Link)`
    padding: 0.5rem 1rem;
    font-size: 1rem;
    background: #fff;
    color: #000;
    border: none;
    cursor: pointer;
    border-radius: 20px;
    text-decoration: none; // Prevent underline on links
    transition: background 0.3s ease, color 0.3s ease; // Add transition for background and color

    &:hover {
        background: #000; // Change background to black on hover
        color: #fff; // Change text color to white on hover
    }
`;

const HeroImage = styled.img`
    width: 50%;
    height: auto;

    @media (max-width: 768px) {
        display: none; // Hide image on mobile
    }
`;

const LogoContainer = styled.div`
    display: flex;
    justify-content: space-between; // Space logos horizontally on desktop
    align-items: center; // Align logos to center
    width: 100%;
    margin-top: 3rem; // Margin for spacing

    @media (max-width: 768px) {
        flex-direction: column; // Stack logos vertically on mobile
    }
`;

const LogoBox = styled.div`
    border-radius: 10px;
    padding: 20px;
    background: transparent;
    border: 2px solid rgba(255, 255, 255, 0.3);
    width: 120px;
    height: 120px;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease; // Smooth hover transition

    &:hover {
        transform: scale(1.1); // Enlarge on hover
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); // Add shadow on hover
        border-color: rgba(255, 255, 255, 0.5); // Slightly brighter border
    }

    @media (max-width: 768px) {
        margin: 1rem 0;
    }
`;

const Logo = styled.img`
    max-width: 90%; // Maintain aspect ratio
    max-height: 90%; // Maintain aspect ratio

    /* Custom styles for AWS and ESM logos */
    ${props => props.large && `
        max-width: 140%; // Make these logos bigger
        max-height: 140%; // Scale proportionally
    `}
`;

const Footer = styled.footer`
    text-align: center;
    padding: 0.5rem 0;
    background: transparent;
    color: #fff;
    width: 100%;
    margin-top: auto;
`;

const FooterP = styled.p`
    margin-top: -35px; // Adjust margin to move footer higher
`;

const StatsSection = styled.section`
    display: flex;
    justify-content: space-between; // Space items evenly
    align-items: center; // Center items vertically
    margin: 2rem 0; // Margin for spacing
    width: 100%; // Full width
    padding: 1rem 0; // Padding for the section
`;

const StatBox = styled.div`
    flex: 1; // Equal space for each box
    text-align: center; // Center text in each box

    @media (max-width: 768px) {
        margin: 1rem 0; // Space out items on mobile
    }
`;

const StatNumber = styled.h2`
    font-size: 3rem; // Large font for the number
    margin: 0; // Remove default margin
    color: #1c5ee1; // Color for the number
`;

const StatCaption = styled.p`
    font-size: 1rem; // Font size for the caption
    margin: 0; // Remove default margin
`;

const Home = () => {
    const [isNavOpen, setIsNavOpen] = useState(false);
    const isMobile = useMediaQuery({ query: '(max-width: 768px)' });

    const toggleNav = () => {
        setIsNavOpen(!isNavOpen);
    };

    const closeNav = () => {
        setIsNavOpen(false);
    };

    return (
        <AppContainer>
            <Header>
    <NavToggle onClick={toggleNav}>
        &#9776; {/* Hamburger Icon */}
    </NavToggle>
    <a href="/">
        <img src="/CHOFormer_logo.png" alt="CHOFormer Logo" style={{ height: '40px', marginRight: '20px' }} />
    </a>
    <nav>
        <NavUl isOpen={isNavOpen}>
            {isMobile && (
                <CloseButton onClick={closeNav}>
                    &times; {/* Close Icon */}
                </CloseButton>
            )}
            <NavLi><NavA href="/">Home</NavA></NavLi>
            <NavLi><NavA href="/CHOFormer">CHOFormer</NavA></NavLi>
            <NavLi><NavA href="/choexp">CHOExp</NavA></NavLi>
            <NavLi><NavA href="/about">About</NavA></NavLi>
        </NavUl>
    </nav>
</Header>

            <HeroContainer>
                <Hero>
                    <HeroH1>CHOFormer</HeroH1>
                    <HeroP>Optimizing Protein Expression in CHO Cells</HeroP>
                    <HeroP>From Months to Minutes.</HeroP>
                    <HeroButton to="/CHOFormer" style={{ marginRight: '1rem' }}>CHOFormer</HeroButton>
                    <HeroButton to="/choexp">CHOExp</HeroButton>
                    
                    {/* Logo section */}
                    <LogoContainer>
                        <LogoBox>
                            <Logo src={ncbiLogo} alt="NCBI Logo" />
                        </LogoBox>
                        <LogoBox>
                            <Logo src={openaiLogo} alt="OpenAI Logo" />
                        </LogoBox>
                        <LogoBox
                        
                        style={{ position: 'relative' }} 
                        onMouseEnter={(e) => {
                            const text = e.currentTarget.querySelector('.logo-text');
                            if (text) {
                                text.style.visibility = 'visible';
                                text.style.opacity = '1';
                            }
                        }} 
                        onMouseLeave={(e) => {
                            const text = e.currentTarget.querySelector('.logo-text');
                            if (text) {
                                text.style.visibility = 'hidden';
                                text.style.opacity = '0';
                            }
                        }}
                        
                        >
                            <Logo src={awsLogo} large alt="AWS Logo" />
                            <span 
                                style={{ 
                                    position: 'absolute', 
                                    bottom: '-25px', 
                                    left: '50%', 
                                    transform: 'translateX(-50%)', 
                                    visibility: 'hidden', 
                                    opacity: 0, 
                                    transition: 'visibility 0s, opacity 0.2s ease', 
                                    color: '#fff' 
                                }} 
                                className="logo-text"
                            >
                                EC2
                            </span>
                        </LogoBox>
                        <LogoBox 
                            style={{ position: 'relative' }} 
                            onMouseEnter={(e) => {
                                const text = e.currentTarget.querySelector('.logo-text');
                                if (text) {
                                    text.style.visibility = 'visible';
                                    text.style.opacity = '1';
                                }
                            }} 
                            onMouseLeave={(e) => {
                                const text = e.currentTarget.querySelector('.logo-text');
                                if (text) {
                                    text.style.visibility = 'hidden';
                                    text.style.opacity = '0';
                                }
                            }}
                        >
                            <Logo 
                                src={esmlogo} 
                                alt="ESM Logo" 
                                onClick={() => window.open('https://evolutionaryscale.ai/#', '_blank')} 
                                style={{ cursor: 'pointer' }} 
                            />
                            <span 
                                style={{ 
                                    position: 'absolute', 
                                    bottom: '-25px', 
                                    left: '50%', 
                                    transform: 'translateX(-50%)', 
                                    visibility: 'hidden', 
                                    opacity: 0, 
                                    transition: 'visibility 0s, opacity 0.2s ease', 
                                    color: '#fff' 
                                }} 
                                className="logo-text"
                            >
                                ESM-2-650M
                            </span>
                        </LogoBox>
                    </LogoContainer>

                    {/* Stats Section */}
                    <StatsSection>
                        <StatBox>
                            <StatNumber>96.98%</StatNumber>
                            <StatCaption>Improved Protein Expression</StatCaption>
                        </StatBox>
                        <StatBox>
                            <StatNumber>2452×</StatNumber>
                            <StatCaption>Increase in Protein Expression</StatCaption>
                        </StatBox>
                        <StatBox>
                            <StatNumber>0.847</StatNumber>
                            <StatCaption>Mean Codon Adaptation Index</StatCaption>
                        </StatBox>
                    </StatsSection>
                </Hero>
                {isMobile ? null : <HeroImage src={gene} alt="Gene" />}
            </HeroContainer>
            <Footer>
                <FooterP>&copy; 2024 CHOFormer. All rights reserved.</FooterP>
            </Footer>
        </AppContainer>
    );
};

export default Home;