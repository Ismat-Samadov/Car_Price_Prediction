/* app.js */
document.addEventListener('DOMContentLoaded', function() {
  const form = document.querySelector('form');
  form.addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(form);
    const json = {};
    formData.forEach((value, key) => {
      json[key] = value;
    });

    fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(json)
    })
      .then(response => response.json())
      .then(data => {
        const resultDiv = document.getElementById('prediction-result');
        if (data.prediction) {
          resultDiv.innerHTML = `<h2>Predicted Price: ${data.prediction}</h2>`;
        } else if (data.error) {
          resultDiv.innerHTML = `<h2>Error: ${data.error}</h2>`;
        }
      })
      .catch(error => {
        console.error('Error:', error);
      });
  });
});