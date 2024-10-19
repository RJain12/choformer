'use client'

import Image from 'next/image';
import Link from 'next/link';
import { useState, useEffect } from 'react';

// Import images
import geneImage from '@/public/gene.png';
import ncbiLogo from '@/public/ncbi.png';
import openaiLogo from '@/public/openai.png';
import awsLogo from '@/public/aws.png';
import esmLogo from '@/public/esm.png';

const Home = () => {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const toggleNav = () => {
    setIsNavOpen(!isNavOpen);
  };

  const closeNav = () => {
    setIsNavOpen(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-start px-4 md:px-8 text-left text-white bg-gradient-radial from-blue-600 via-blue-900 to-black">
      <header className="w-full flex justify-between items-center py-4">
        <div className="md:hidden text-2xl cursor-pointer" onClick={toggleNav}>
          &#9776;
        </div>
        <nav className={`${isMobile ? 'fixed inset-0 bg-gray-900 z-50' : ''} ${isNavOpen ? 'block' : 'hidden md:block'}`}>
          {isMobile && (
            <div className="text-right p-4 text-2xl cursor-pointer" onClick={closeNav}>
              &times;
            </div>
          )}
          <ul className={`${isMobile ? 'flex flex-col items-center space-y-4 pt-8' : 'flex space-x-4'}`}>
            <li><Link href="/" className="hover:bg-white hover:bg-opacity-20 px-3 py-2 rounded">Home</Link></li>
            <li><Link href="/CHOFormer" className="hover:bg-white hover:bg-opacity-20 px-3 py-2 rounded">CHOFormer</Link></li>
            <li><Link href="/choexp" className="hover:bg-white hover:bg-opacity-20 px-3 py-2 rounded">CHOExp</Link></li>
            <li><Link href="/about" className="hover:bg-white hover:bg-opacity-20 px-3 py-2 rounded">About</Link></li>
          </ul>
        </nav>
      </header>

      <main className="flex flex-col md:flex-row justify-between items-start w-full mt-8">
        <section className="w-full md:w-2/3">
          <h1 className="text-5xl mb-4">CHOFormer</h1>
          <p className="text-xl mb-8">Some text here.</p>
          <div className="space-x-4">
            <Link href="/CHOFormer" className="bg-white text-black px-4 py-2 rounded-full hover:bg-black hover:text-white transition-colors">
              CHOFormer
            </Link>
            <Link href="/choexp" className="bg-white text-black px-4 py-2 rounded-full hover:bg-black hover:text-white transition-colors">
              CHOExp
            </Link>
          </div>

          <div className="flex justify-between mt-12 space-x-4">
            {[ncbiLogo, openaiLogo, awsLogo, esmLogo].map((logo, index) => (
              <div key={index} className="border-2 border-white border-opacity-30 rounded-lg p-4 w-24 h-24 flex items-center justify-center hover:scale-110 hover:shadow-lg transition-all">
                <Image src={logo} alt={`Logo ${index + 1}`} width={80} height={80} className="max-w-full max-h-full" />
              </div>
            ))}
          </div>

          <div className="flex justify-between mt-12 space-x-4">
            {[
              { number: '123', caption: 'ABC' },
              { number: '456', caption: 'DEF' },
              { number: '789', caption: 'HIJ' },
            ].map((stat, index) => (
              <div key={index} className="text-center">
                <h2 className="text-4xl text-blue-500">{stat.number}</h2>
                <p>{stat.caption}</p>
              </div>
            ))}
          </div>
        </section>

        {!isMobile && (
          <div className="w-1/3">
            <Image src={geneImage} alt="Gene" layout="responsive" />
          </div>
        )}
      </main>

      <footer className="mt-auto py-2 text-center w-full">
        <p>&copy; 2024 CHOFormer. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default Home;