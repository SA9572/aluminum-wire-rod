document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        // Update results
        document.getElementById('result').textContent = data.result;
        document.getElementById('confidence').textContent = data.confidence;
        document.getElementById('suggestion').textContent = data.suggestion;

        // Generate chart
        const ctx = document.getElementById('qualityChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Tensile Strength', 'Proof Stress', 'Elongation'],
                datasets: [{
                    label: 'Input Values',
                    data: [data.tensile_strength, data.proof_stress, data.elongation],
                    backgroundColor: ['#007BFF', '#28A745', '#FFC107']
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Value' }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error:', error);
    }
});