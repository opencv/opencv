// SPDX-License-Identifier: MIT
/**

Doxygen Awesome
https://github.com/jothepro/doxygen-awesome-css

Copyright (c) 2021 - 2025 jothepro

*/

class DoxygenAwesomeDarkModeToggle extends HTMLElement {
    // SVG icons from https://fonts.google.com/icons
    // Licensed under the Apache 2.0 license:
    // https://www.apache.org/licenses/LICENSE-2.0.html
    static lightModeIcon = `<svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 24 24" height="24px" viewBox="0 0 24 24" width="24px" fill="#FCBF00"><rect fill="none" height="24" width="24"/><circle cx="12" cy="12" opacity=".3" r="3"/><path d="M12,9c1.65,0,3,1.35,3,3s-1.35,3-3,3s-3-1.35-3-3S10.35,9,12,9 M12,7c-2.76,0-5,2.24-5,5s2.24,5,5,5s5-2.24,5-5 S14.76,7,12,7L12,7z M2,13l2,0c0.55,0,1-0.45,1-1s-0.45-1-1-1l-2,0c-0.55,0-1,0.45-1,1S1.45,13,2,13z M20,13l2,0c0.55,0,1-0.45,1-1 s-0.45-1-1-1l-2,0c-0.55,0-1,0.45-1,1S19.45,13,20,13z M11,2v2c0,0.55,0.45,1,1,1s1-0.45,1-1V2c0-0.55-0.45-1-1-1S11,1.45,11,2z M11,20v2c0,0.55,0.45,1,1,1s1-0.45,1-1v-2c0-0.55-0.45-1-1-1C11.45,19,11,19.45,11,20z M5.99,4.58c-0.39-0.39-1.03-0.39-1.41,0 c-0.39,0.39-0.39,1.03,0,1.41l1.06,1.06c0.39,0.39,1.03,0.39,1.41,0s0.39-1.03,0-1.41L5.99,4.58z M18.36,16.95 c-0.39-0.39-1.03-0.39-1.41,0c-0.39,0.39-0.39,1.03,0,1.41l1.06,1.06c0.39,0.39,1.03,0.39,1.41,0c0.39-0.39,0.39-1.03,0-1.41 L18.36,16.95z M19.42,5.99c0.39-0.39,0.39-1.03,0-1.41c-0.39-0.39-1.03-0.39-1.41,0l-1.06,1.06c-0.39,0.39-0.39,1.03,0,1.41 s1.03,0.39,1.41,0L19.42,5.99z M7.05,18.36c0.39-0.39,0.39-1.03,0-1.41c-0.39-0.39-1.03-0.39-1.41,0l-1.06,1.06 c-0.39,0.39-0.39,1.03,0,1.41s1.03,0.39,1.41,0L7.05,18.36z"/></svg>`
    static darkModeIcon = `<svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 24 24" height="24px" viewBox="0 0 24 24" width="24px" fill="#FE9700"><rect fill="none" height="24" width="24"/><path d="M9.37,5.51C9.19,6.15,9.1,6.82,9.1,7.5c0,4.08,3.32,7.4,7.4,7.4c0.68,0,1.35-0.09,1.99-0.27 C17.45,17.19,14.93,19,12,19c-3.86,0-7-3.14-7-7C5,9.07,6.81,6.55,9.37,5.51z" opacity=".3"/><path d="M9.37,5.51C9.19,6.15,9.1,6.82,9.1,7.5c0,4.08,3.32,7.4,7.4,7.4c0.68,0,1.35-0.09,1.99-0.27C17.45,17.19,14.93,19,12,19 c-3.86,0-7-3.14-7-7C5,9.07,6.81,6.55,9.37,5.51z M12,3c-4.97,0-9,4.03-9,9s4.03,9,9,9s9-4.03,9-9c0-0.46-0.04-0.92-0.1-1.36 c-0.98,1.37-2.58,2.26-4.4,2.26c-2.98,0-5.4-2.42-5.4-5.4c0-1.81,0.89-3.42,2.26-4.4C12.92,3.04,12.46,3,12,3L12,3z"/></svg>`
    static title = "Toggle Light/Dark Mode"

    static prefersLightModeInDarkModeKey = "prefers-light-mode-in-dark-mode"
    static prefersDarkModeInLightModeKey = "prefers-dark-mode-in-light-mode"

    static _staticConstructor = function() {
        DoxygenAwesomeDarkModeToggle.enableDarkMode(DoxygenAwesomeDarkModeToggle.userPreference)
        // Update the color scheme when the browsers preference changes
        // without user interaction on the website.
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
            DoxygenAwesomeDarkModeToggle.onSystemPreferenceChanged()
        })
        // Update the color scheme when the tab is made visible again.
        // It is possible that the appearance was changed in another tab
        // while this tab was in the background.
        document.addEventListener("visibilitychange", visibilityState => {
            if (document.visibilityState === 'visible') {
                DoxygenAwesomeDarkModeToggle.onSystemPreferenceChanged()
            }
        });
    }()

    static init() {
        $(function() {
            $(document).ready(function() {
                const toggleButton = document.createElement('doxygen-awesome-dark-mode-toggle')
                toggleButton.title = DoxygenAwesomeDarkModeToggle.title
                toggleButton.updateIcon()

                window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
                    toggleButton.updateIcon()
                })
                document.addEventListener("visibilitychange", visibilityState => {
                    if (document.visibilityState === 'visible') {
                        toggleButton.updateIcon()
                    }
                });

                $(document).ready(function(){
                    document.getElementById("MSearchBox").parentNode.appendChild(toggleButton)
                })
                $(window).resize(function(){
                    document.getElementById("MSearchBox").parentNode.appendChild(toggleButton)
                })
            })
        })
    }

    constructor() {
        super();
        this.onclick=this.toggleDarkMode
    }

    /**
     * @returns `true` for dark-mode, `false` for light-mode system preference
     */
    static get systemPreference() {
        return window.matchMedia('(prefers-color-scheme: dark)').matches
    }

    /**
     * @returns `true` for dark-mode, `false` for light-mode user preference
     */
    static get userPreference() {
        return (!DoxygenAwesomeDarkModeToggle.systemPreference && localStorage.getItem(DoxygenAwesomeDarkModeToggle.prefersDarkModeInLightModeKey)) ||
        (DoxygenAwesomeDarkModeToggle.systemPreference && !localStorage.getItem(DoxygenAwesomeDarkModeToggle.prefersLightModeInDarkModeKey))
    }

    static set userPreference(userPreference) {
        DoxygenAwesomeDarkModeToggle.darkModeEnabled = userPreference
        if(!userPreference) {
            if(DoxygenAwesomeDarkModeToggle.systemPreference) {
                localStorage.setItem(DoxygenAwesomeDarkModeToggle.prefersLightModeInDarkModeKey, true)
            } else {
                localStorage.removeItem(DoxygenAwesomeDarkModeToggle.prefersDarkModeInLightModeKey)
            }
        } else {
            if(!DoxygenAwesomeDarkModeToggle.systemPreference) {
                localStorage.setItem(DoxygenAwesomeDarkModeToggle.prefersDarkModeInLightModeKey, true)
            } else {
                localStorage.removeItem(DoxygenAwesomeDarkModeToggle.prefersLightModeInDarkModeKey)
            }
        }
        DoxygenAwesomeDarkModeToggle.onUserPreferenceChanged()
    }

    static enableDarkMode(enable) {
        if(enable) {
            DoxygenAwesomeDarkModeToggle.darkModeEnabled = true
            document.documentElement.classList.add("dark-mode")
            document.documentElement.classList.remove("light-mode")
        } else {
            DoxygenAwesomeDarkModeToggle.darkModeEnabled = false
            document.documentElement.classList.remove("dark-mode")
            document.documentElement.classList.add("light-mode")
        }
    }

    static onSystemPreferenceChanged() {
        DoxygenAwesomeDarkModeToggle.darkModeEnabled = DoxygenAwesomeDarkModeToggle.userPreference
        DoxygenAwesomeDarkModeToggle.enableDarkMode(DoxygenAwesomeDarkModeToggle.darkModeEnabled)
    }

    static onUserPreferenceChanged() {
        DoxygenAwesomeDarkModeToggle.enableDarkMode(DoxygenAwesomeDarkModeToggle.darkModeEnabled)
    }

    toggleDarkMode() {
        DoxygenAwesomeDarkModeToggle.userPreference = !DoxygenAwesomeDarkModeToggle.userPreference
        this.updateIcon()
    }

    updateIcon() {
        if(DoxygenAwesomeDarkModeToggle.darkModeEnabled) {
            this.innerHTML = DoxygenAwesomeDarkModeToggle.darkModeIcon
        } else {
            this.innerHTML = DoxygenAwesomeDarkModeToggle.lightModeIcon
        }
    }
}

customElements.define("doxygen-awesome-dark-mode-toggle", DoxygenAwesomeDarkModeToggle);
