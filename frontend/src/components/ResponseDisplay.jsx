import React from 'react';

const ResponseDisplay = ({ response, responseType }) => {
  const renderResponse = () => {
    if (typeof response === 'object' && response.aiResponse) {
      return (
        <div>
          <strong>AI Response:</strong><br />
          {response.aiResponse}<br /><br />
          <strong>Thread ID:</strong> {response.threadId}
        </div>
      );
    }
    return response;
  };

  return (
    <div className={`response ${responseType}`}>
      <h3>Response:</h3>
      <div className="response-content">
        {renderResponse()}
      </div>
    </div>
  );
};

export default ResponseDisplay;
