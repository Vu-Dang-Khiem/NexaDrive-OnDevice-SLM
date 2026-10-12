import "@fontsource/be-vietnam-pro/400.css";
import "@fontsource/be-vietnam-pro/500.css";
import "@fontsource/be-vietnam-pro/600.css";
import "@fontsource/be-vietnam-pro/700.css";
import "@fontsource/jetbrains-mono/400.css";
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

// font đóng gói sẵn trong dự án (không tải từ Google Fonts) -> demo chạy được khi mất mạng
const el = document.getElementById("root");
if (!el) throw new Error("không tìm thấy #root");

document.body.style.margin = "0";
createRoot(el).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
