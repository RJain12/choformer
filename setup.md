## Setup

### Frontend

- Install dependencies: `npm install`
- Update `API_URL` in `web/src/constants.js`(do not add a trailing slash)
- Start frontend: `npm start`

#### Hosting on vercel
- Update the API_URL and use vercel's github integration to deploy. It will automatically setup rest of the things.

### Backend

- Install dependencies: `pip install -r requirements.txt`
- Start backend: `python app.py`

The backend will by default run on port 8000
Replace gpt2 w cho