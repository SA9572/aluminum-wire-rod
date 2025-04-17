// document.getElementById('predictionForm').addEventListener('submit', async (e) => {
//     e.preventDefault();

//     const formData = new FormData(e.target);
    
//     try {
//         const response = await fetch('/predict', {
//             method: 'POST',
//             body: formData
//         });
//         const data = await response.json();

//         if (data.error) {
//             alert('Error: ' + data.error);
//             return;
//         }

//         // Update results
//         document.getElementById('result').textContent = data.result;
//         document.getElementById('confidence').textContent = data.confidence;
//         document.getElementById('suggestion').textContent = data.suggestion;

//         // Generate chart
//         const ctx = document.getElementById('qualityChart').getContext('2d');
//         new Chart(ctx, {
//             type: 'bar',
//             data: {
//                 labels: ['Tensile Strength', 'Proof Stress', 'Elongation'],
//                 datasets: [{
//                     label: 'Input Values',
//                     data: [data.tensile_strength, data.proof_stress, data.elongation],
//                     backgroundColor: ['#007BFF', '#28A745', '#FFC107']
//                 }]
//             },
//             options: {
//                 scales: {
//                     y: {
//                         beginAtZero: true,
//                         title: { display: true, text: 'Value' }
//                     }
//                 }
//             }
//         });
//     } catch (error) {
//         console.error('Error:', error);
//     }
// });


// document.getElementById('predict-form').addEventListener('submit', async function (e) {
//     e.preventDefault();

//     const formData = new FormData(this);
//     const data = {
//         tensile_strength: formData.get('tensile_strength'),
//         proof_stress: formData.get('proof_stress'),
//         elongation: formData.get('elongation')
//     };

//     try {
//         const response = await fetch('/predict', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/x-www-form-urlencoded'
//             },
//             body: new URLSearchParams(data)
//         });

//         const result = await response.json();
//         const resultDiv = document.getElementById('result');

//         if (response.ok) {
//             resultDiv.innerHTML = `
//                 <p class="text-green-600 font-semibold">${result.result}</p>
//                 <p>Confidence: ${result.confidence}</p>
//                 <p>Suggestion: ${result.suggestion}</p>
//                 <p>Input: Tensile Strength = ${result.tensile_strength} MPa, Proof Stress = ${result.proof_stress} MPa, Elongation = ${result.elongation}%</p>
//             `;
//         } else {
//             resultDiv.innerHTML = `<p class="text-red-600">Error: ${result.error}</p>`;
//         }
//     } catch (error) {
//         document.getElementById('result').innerHTML = `<p class="text-red-600">Error: Failed to connect to the server</p>`;
//     }
// });

// Prediction Form
const predictForm = document.getElementById('predict-form');
if (predictForm) {
    predictForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        const formData = new FormData(this);
        const data = {
            tensile_strength: formData.get('tensile_strength'),
            proof_stress: formData.get('proof_stress'),
            elongation: formData.get('elongation')
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: new URLSearchParams(data)
            });

            const result = await response.json();
            const resultDiv = document.getElementById('result');

            if (response.ok) {
                resultDiv.innerHTML = `
                    <p class="text-green-600 font-semibold">${result.result}</p>
                    <p>Confidence: ${result.confidence}</p>
                    <p>Suggestion: ${result.suggestion}</p>
                    <p>Input: Tensile Strength = ${result.tensile_strength} MPa, Proof Stress = ${result.proof_stress} MPa, Elongation = ${result.elongation}%</p>
                `;
            } else {
                resultDiv.innerHTML = `<p class="text-red-600">Error: ${result.error}</p>`;
            }
        } catch (error) {
            document.getElementById('result').innerHTML = `<p class="text-red-600">Error: Failed to connect to the server</p>`;
        }
    });
}

// Contact Form (Mock submission)
const contactForm = document.getElementById('contact-form');
if (contactForm) {
    contactForm.addEventListener('submit', function (e) {
        e.preventDefault();
        const resultDiv = document.getElementById('contact-result');
        resultDiv.innerHTML = `<p class="text-green-600">Thank you for your message! We'll get back to you soon.</p>`;
        contactForm.reset();
    });
}

document.addEventListener('DOMContentLoaded', function () {
    const predictForm = document.getElementById('predict-form');
    const predictButton = document.getElementById('predict-button');
    const resultDiv = document.getElementById('result');
    const validationError = document.getElementById('validation-error');

    if (predictForm && predictButton && resultDiv && validationError) {
        predictForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const formData = new FormData(this);
            const data = {
                tensile_strength: formData.get('tensile_strength'),
                proof_stress: formData.get('proof_stress'),
                elongation: formData.get('elongation')
            };

            // Reset validation error
            validationError.classList.add('hidden');
            validationError.textContent = '';

            // Validate inputs
            if (!data.tensile_strength || !data.proof_stress || !data.elongation) {
                validationError.textContent = 'All fields are required.';
                validationError.classList.remove('hidden');
                return;
            }

            const tensile = parseFloat(data.tensile_strength);
            const proof = parseFloat(data.proof_stress);
            const elongation = parseFloat(data.elongation);

            if (isNaN(tensile) || isNaN(proof) || isNaN(elongation)) {
                validationError.textContent = 'All inputs must be valid numbers.';
                validationError.classList.remove('hidden');
                return;
            }

            if (tensile <= 0 || proof <= 0 || elongation <= 0) {
                validationError.textContent = 'All inputs must be positive numbers.';
                validationError.classList.remove('hidden');
                return;
            }

            // Show loading state
            predictButton.disabled = true;
            predictButton.textContent = 'Predicting...';
            resultDiv.innerHTML = '<p class="text-gray-600">Processing...</p>';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams(data)
                });

                const result = await response.json();

                if (response.ok) {
                    resultDiv.innerHTML = `
                        <p class="text-green-600 font-semibold">${result.result}</p>
                        <p>Confidence: <span class="text-blue-400">${result.confidence}</span></p>
                        <p>Suggestion: <span class="text-gray-300">${result.suggestion}</span></p>
                        <p>Input: Tensile Strength = ${result.tensile_strength} MPa, Proof Stress = ${result.proof_stress} MPa, Elongation = ${result.elongation}%</p>
                    `;
                } else {
                    resultDiv.innerHTML = `<p class="text-red-600">Error: ${result.error}</p>`;
                }
            } catch (error) {
                resultDiv.innerHTML = '<p class="text-red-600">Error: Failed to connect to the server. Please ensure the server is running.</p>';
            } finally {
                predictButton.disabled = false;
                predictButton.textContent = 'Predict Quality';
            }
        });
    }
});

