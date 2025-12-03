# ChatBot API Tester - React Frontend

A React application built with Vite for testing a ChatBot API. This application provides a user-friendly interface to interact with a FastAPI chatbot backend.

## Features

- **Send Messages**: Send messages to the chatbot and receive AI responses
- **Chat History**: View conversation history for any thread
- **Multiple Threads**: Manage multiple conversation threads with different IDs
- **Reset Conversations**: Clear conversation history for any thread
- **Real-time UI**: Responsive interface with loading states and error handling

## Technical Stack

- **Frontend**: React 18 with Vite
- **Styling**: CSS modules
- **HTTP Client**: Fetch API
- **Backend API**: FastAPI server (expected at http://localhost:8001)

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- FastAPI backend server running on port 8001

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

### Backend Requirements

Make sure your FastAPI backend is running on `http://localhost:8001` with the following endpoints:

- `POST /chat?thread_id={id}` - Send a message to the chatbot
- `GET /chat/history/{thread_id}` - Get conversation history
- `POST /chat/reset/{thread_id}` - Reset a conversation thread

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Integration

The application connects to a FastAPI backend and expects the following request/response format:

### Send Message
```javascript
POST /chat?thread_id=default
{
  "text": "Hello, how are you?"
}
```

Response:
```javascript
{
  "response": "I'm doing well, thank you!",
  "thread_id": "default"
}
```

### Get History
```javascript
GET /chat/history/default
```

Response:
```javascript
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ],
  "thread_id": "default"
}
```

## Project Structure

```
src/
├── components/
│   ├── ChatBot.jsx          # Main chat interface component
│   ├── ChatBot.css          # Styling for chat interface
│   ├── MessageForm.jsx      # Form for sending messages
│   ├── ResponseDisplay.jsx  # Display API responses
│   └── ChatHistory.jsx      # Display chat history
├── App.jsx                  # Main app component
├── App.css                  # App-level styling
├── index.css                # Global styles
└── main.jsx                 # Entry point
```
