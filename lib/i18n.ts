"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import en from "../dictionaries/en.json";
import ta from "../dictionaries/ta.json";

export type Locale = "en" | "ta";

const dictionaries = {
  en,
  ta,
};

type DictionaryType = typeof en;
export type TranslationKey = keyof DictionaryType;

interface I18nContextProps {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: TranslationKey) => string;
}

const I18nContext = createContext<I18nContextProps | undefined>(undefined);

/**
 * I18nProvider provides client-side internationalization for English and Tamil.
 * It stores and manages the current locale state, persisting it in local storage,
 * and exposes a translation utility function `t` to dynamically translate keys.
 * 
 * Uses React.createElement to keep the file as a standard .ts file rather than .tsx.
 */
export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  // Load language preference from localStorage on mount (client-side only)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedLocale = localStorage.getItem("edgetriage_locale") as Locale;
      if (savedLocale === "en" || savedLocale === "ta") {
        setLocaleState(savedLocale);
      }
    }
  }, []);

  const setLocale = (newLocale: Locale) => {
    setLocaleState(newLocale);
    if (typeof window !== "undefined") {
      localStorage.setItem("edgetriage_locale", newLocale);
    }
  };

  const t = (key: TranslationKey): string => {
    const dictionary = dictionaries[locale] || dictionaries.en;
    // Fallback chain: locale-specific dictionary -> English dictionary -> raw key
    return (dictionary[key] || en[key] || key) as string;
  };

  return React.createElement(
    I18nContext.Provider,
    { value: { locale, setLocale, t } },
    children
  );
}

/**
 * Custom React hook to consume translation strings and toggle locales in client components.
 */
export function useTranslation() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useTranslation must be used within an I18nProvider");
  }
  return context;
}
