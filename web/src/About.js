import React, { useState } from 'react';
import { useMediaQuery } from 'react-responsive';
import styled, { keyframes, createGlobalStyle } from 'styled-components';
import { motion } from 'framer-motion';
import { ChevronRight, Menu, X } from 'lucide-react';
import CHOFormerlogo from './assets/CHOFormer_logo_light.png'
import CHOFormerlogo_notagline from './assets/CHOFormer_logo_light_notagline.png'

// Assume these imports are correct
import ncbiLogo from './assets/ncbi.png';
import openaiLogo from './assets/openai.png';
import awsLogo from './assets/aws.png';

const GlobalStyle = createGlobalStyle`
  body {
    margin: 0;
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

const About = () => {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const isMobile = useMediaQuery({ query: '(max-width: 768px)' });

  const toggleNav = () => setIsNavOpen(!isNavOpen);

  const navVariants = {
    open: { x: 0 },
    closed: { x: '100%' },
  };

  return (
    <>
      <GlobalStyle />
      <AppContainer>
        <Header>
        <a href="/">
        <img src={CHOFormerlogo_notagline} alt="CHOFormer Logo" style={{ height: '60px', marginRight: '20px' }} />
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
        <ContentContainer>
          <Content>
            <Title
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              About CHOFormer
            </Title>
            <Paragraph
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              CHOFormer is a cutting-edge <Highlight>Transformer decoder model</Highlight> developed to optimize codon sequences for <Highlight>improving protein expression</Highlight> in Chinese Hamster Ovary (CHO) cells. It addresses the challenges posed by the genetic code's degeneracy, where 61 sense codons encode only 20 standard amino acids.
            </Paragraph>
            <Paragraph
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              While synonymous codons encode the same amino acid, their selection can drastically influence the <Highlight>speed and accuracy</Highlight> of protein production. CHOFormer leverages advanced machine learning techniques to optimize codon selection based on the relationship between protein expression and codon usage patterns.
            </Paragraph>
            <Paragraph
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              As CHO cells are used in the production of nearly 70% of recombinant pharmaceuticals, including monoclonal antibodies and other therapeutic proteins, optimizing protein yield in these cells is critical for drug development. CHOFormer significantly improves protein yields, with <Highlight>96.98% of proteins showing higher expression</Highlight> when optimized with our model.
            </Paragraph>
            <Paragraph
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.8 }}
            >
              Perhaps most importantly, CHOFormer reduces the optimization process from months in laboratory settings to <Highlight>mere minutes</Highlight>, drastically accelerating the drug manufacturing process and potentially bringing life-saving treatments to patients faster.
            </Paragraph>
          </Content>
          <LogoContainer>
            <Logo
              src={ncbiLogo}
              alt="NCBI Logo"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 1 }}
            />
            <Logo
              src={openaiLogo}
              alt="OpenAI Logo"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 1.2 }}
            />
            <Logo
              src={awsLogo}
              alt="AWS Logo"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 1.4 }}
            />
          </LogoContainer>
        </ContentContainer>
        <Footer>
          <p>&copy; 2024 CHOFormer. All rights reserved.</p>
        </Footer>
      </AppContainer>
    </>
  );
};

export default About;