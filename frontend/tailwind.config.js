/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        graphite: {
          950: "#0F1216",
          900: "#14171C",
          800: "#1C2028",
          700: "#262B35",
          600: "#2E343F",
          border: "#2A2F3A",
        },
        ink: {
          100: "#ECEAE5",
          300: "#C7C9CE",
          500: "#8B8F98",
          700: "#858A93",
        },
        signal: {
          amber: "#E8A33D",
          amberDim: "#8A6428",
          green: "#4FAE8C",
          red: "#D9695F",
        },
      },
      fontFamily: {
        display: ["Space Grotesk", "system-ui", "sans-serif"],
        body: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
