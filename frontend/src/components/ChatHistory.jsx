import React from 'react';

const ChatHistory = ({ messages, threadId }) => {
  const renderMessages = () => {
    if (messages.length === 0) {
      return <p>No messages in this thread yet.</p>;
    }

    return messages.map((msg, index) => (
      <div 
        key={index} 
        className={`message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`}
      >
        <strong>{msg.role === 'user' ? 'User' : 'Assistant'}:</strong> {msg.content}
      </div>
    ));
  };

  return (
    <div className="chat-history">
      <h3>Chat History:</h3>
      <div className="history-content">
        {renderMessages()}
      </div>
    </div>
  );
};

export default ChatHistory;
