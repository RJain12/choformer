import React, { useState } from 'react';
import { useMediaQuery } from 'react-responsive';
import styled, { keyframes } from 'styled-components';
import ncbiLogo from './assets/ncbi.png'; // Adjust the path as necessary
import openaiLogo from './assets/openai.png'; // Adjust the path as necessary
import awsLogo from './assets/aws.png'; // Adjust the path as necessary

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
            @media (min-width: 769px) {
        height: 100vh; // Ensure the container takes full viewport height on desktop
        overflow: hidden; // Prevent scrolling on desktop
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
        max-height: 500px;
        opacity: 1;
    }
`;

const slideUp = keyframes`
    from {
        max-height: 500px;
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
    overflow: hidden;
    display: flex;
    @media (max-width: 768px) {
        display: block; // Use block display for mobile
        background-color: #1a1a1a; // Dark background for mobile navbar
        position: fixed; // Fixed position
        top: 0; // Start from the top of the screen
        left: 0;
        right: 0; // Full width
        transition: opacity 0.3s ease;
        border-radius: 8px; // Slightly rounded corners
        z-index: 1000; // Ensure it appears on top
        padding: 1rem 0; // Padding for mobile navbar
        animation: ${props => (props.isOpen ? slideDown : slideUp)} 0.3s ease forwards;
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

const ContentContainer = styled.div`
    display: flex;
    flex-direction: row; // Change to row for larger screens
    align-items: flex-start;
    justify-content: space-between; // Space between text and logos
    flex: 1;
    width: 100%;
    padding: 0 1rem; // Add padding for mobile
    @media (max-width: 768px) {
        flex-direction: column; // Stack items vertically on mobile
        align-items: center; // Center align items
    }
`;

const Content = styled.section`
    text-align: left;
    padding: 2rem 0;
    margin: 0 0 23rem 0;
    background: transparent;
    width: 100%; // Full width on mobile
    max-width: 800px; // Maximum width for larger screens
`;

const ContentH1 = styled.h1`
    margin: 0 0 0.5rem 0;
    margin-top: 20px;
    font-size: 3rem; // Adjusted for responsiveness
`;

const ContentP = styled.p`
    font-size: 1.2rem; // Slightly decrease font size
    margin: 0 0 2rem 0;
`;

const Footer = styled.footer`
    text-align: center;
    padding: 0.5rem 0;
    background: transparent;
    color: #fff;
    width: 100%;
    margin-top: -150rem;
`;

const FooterP = styled.p`
    margin-top: -35px;
`;

const LogoContainer = styled.div`
    display: flex;
    flex-direction: column; // Align logos vertically
    justify-content: flex-start;
    align-items: flex-start; // Align logos to the left
    margin-top: 2rem;
    @media (max-width: 768px) {
    margin-top: -25rem;
    margin-right: -5rem;
}
`;

const Logo = styled.img`
    height: 150px; // Increased height
    width: auto; // Maintain aspect ratio
    margin: 1rem 0; // Space between logos
`;

const AwsLogo = styled(Logo)`
    height: 200px; // Larger height for AWS logo
    margin-left: -4.25rem; // Move AWS logo to the left
    margin-top: -10px;
`;

const About = () => {
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
                <nav>
                    <NavUl isOpen={isNavOpen}>
                        {isMobile && (
                            <CloseButton onClick={closeNav}>
                                &times; {/* Close Icon */}
                            </CloseButton>
                        )}
                        <NavLi><NavA href="/">Home</NavA></NavLi>                       
                        <NavLi><NavA href="/CHOFormer">CHOFormer</NavA></NavLi>
                        <NavLi><NavA href="/CHOExp">CHOExp</NavA></NavLi>
                        <NavLi><NavA href="/about">About</NavA></NavLi>
                    </NavUl>
                </nav>
            </Header>
            <ContentContainer>
                <Content>
                    <ContentH1>About Us</ContentH1>
                    <ContentP>Some more text goes here.</ContentP>
                </Content>
                <LogoContainer>
                    <Logo src={ncbiLogo} alt="NCBI Logo" />
                    <Logo src={openaiLogo} alt="OpenAI Logo" />
                    <AwsLogo src={awsLogo} alt="AWS Logo" />
                </LogoContainer>
            </ContentContainer>
            <Footer>
                <FooterP>&copy; 2024 Dhruv Ramu. All rights reserved.</FooterP>
            </Footer>
        </AppContainer>
    );
};

export default About;
