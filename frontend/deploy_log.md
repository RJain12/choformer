3:13:38 PM: Netlify Build                                                 
3:13:38 PM: ────────────────────────────────────────────────────────────────
3:13:38 PM: ​
3:13:38 PM: ❯ Version
3:13:38 PM:   @netlify/build 29.55.0
3:13:38 PM: ​
3:13:38 PM: ❯ Flags
3:13:38 PM:   accountId: 62bdc81ab4feee368620dfce
3:13:38 PM:   baseRelDir: true
3:13:38 PM:   buildId: 6708f30aeb72c61da7c1c63f
3:13:38 PM:   deployId: 6708f30aeb72c61da7c1c641
3:13:39 PM: ​
3:13:39 PM: ❯ Current directory
3:13:39 PM:   /opt/build/repo
3:13:39 PM: ​
3:13:39 PM: ❯ Config file
3:13:39 PM:   No config file was defined: using default values.
3:13:39 PM: ​
3:13:39 PM: ❯ Context
3:13:39 PM:   production
3:13:39 PM: ​
3:13:39 PM: Build command from Netlify app                                
3:13:39 PM: ────────────────────────────────────────────────────────────────
3:13:39 PM: ​
3:13:39 PM: $ npm run build
3:13:39 PM: > choforma@0.1.0 build
3:13:39 PM: > react-scripts build
3:13:40 PM: Creating an optimized production build...
3:13:44 PM: One of your dependencies, babel-preset-react-app, is importing the
3:13:44 PM: "@babel/plugin-proposal-private-property-in-object" package without
3:13:44 PM: declaring it in its dependencies. This is currently working because
3:13:44 PM: "@babel/plugin-proposal-private-property-in-object" is already in your
3:13:44 PM: node_modules folder for unrelated reasons, but it may break at any time.
3:13:44 PM: babel-preset-react-app is part of the create-react-app project, which
3:13:44 PM: is not maintianed anymore. It is thus unlikely that this bug will
3:13:44 PM: ever be fixed. Add "@babel/plugin-proposal-private-property-in-object" to
3:13:44 PM: your devDependencies to work around this error. This will make this message
3:13:44 PM: go away.
3:13:46 PM: 
3:13:46 PM: Treating warnings as errors because process.env.CI = true.
3:13:46 PM: Most CI servers set it automatically.
3:13:46 PM: 
3:13:46 PM: Failed to compile.
3:13:46 PM: 
3:13:46 PM: [eslint]
3:13:46 PM: src/App.js
3:13:46 PM:   Line 5:8:  'Tool' is defined but never used  no-unused-vars
3:13:46 PM: src/CHOExp.js
3:13:46 PM:   Line 1:27:  'useEffect' is defined but never used  no-unused-vars
3:13:46 PM: src/CHOFormer.js
3:13:46 PM:   Line 1:27:  'useEffect' is defined but never used  no-unused-vars
3:13:46 PM: src/Tool.js
3:13:46 PM:   Line 1:27:  'useEffect' is defined but never used  no-unused-vars
3:13:46 PM: ​
3:13:46 PM: "build.command" failed                                        
3:13:46 PM: ────────────────────────────────────────────────────────────────
3:13:46 PM: ​
3:13:46 PM:   Error message
3:13:46 PM:   Command failed with exit code 1: npm run build (https://ntl.fyi/exit-code-1)
3:13:46 PM: ​
3:13:46 PM:   Error location
3:13:46 PM:   In Build command from Netlify app:
3:13:46 PM:   npm run build
3:13:46 PM: ​
3:13:46 PM:   Resolved config
3:13:46 PM:   build:
3:13:46 PM:     command: npm run build
3:13:46 PM:     commandOrigin: ui
3:13:46 PM:     publish: /opt/build/repo/build
3:13:46 PM:     publishOrigin: ui
3:13:46 PM: Build failed due to a user error: Build script returned non-zero exit code: 2
3:13:47 PM: Failed during stage 'building site': Build script returned non-zero exit code: 2 (https://ntl.fyi/exit-code-2)
3:13:47 PM: Failing build: Failed to build site
3:13:47 PM: Finished processing build request in 29.304s
