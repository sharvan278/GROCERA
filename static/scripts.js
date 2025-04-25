document.addEventListener('DOMContentLoaded', function() {
    // Flash message auto-hide
    const flashes = document.querySelectorAll('.flash');
    if (flashes.length) {
        flashes.forEach(flash => {
            setTimeout(() => {
                flash.style.opacity = '0';
                setTimeout(() => flash.remove(), 300);
            }, 3000);
        });
    }

    // Check if we have uploaded data
    const hasData = sessionStorage.getItem('hasData');
    
    if (hasData) {
        enableDashboard();
        loadSampleData();
    }

    // Handle upload button click
    document.querySelector('.cta-button')?.addEventListener('click', function(e) {
        if (!hasData) {
            e.preventDefault();
            sessionStorage.setItem('hasData', 'true');
            enableDashboard();
            loadSampleData();
        }
    });

    function enableDashboard() {
        // Enable navbar items
        document.querySelectorAll('.nav-disabled').forEach(item => {
            item.classList.remove('nav-disabled');
        });

        // Show action buttons
        const actions = document.getElementById('results-actions');
        if (actions) actions.style.display = 'flex';

        // Make cards clickable
        document.querySelectorAll('.dashboard-card').forEach(card => {
            card.style.cursor = 'pointer';
            card.addEventListener('click', function() {
                const cardId = this.id;
                console.log(`Clicked ${cardId}`);
                // Add your redirection or dynamic loading logic here
            });
        });
    }

    function loadSampleData() {
        // Example dummy data
        const priceContent = document.getElementById('price-content');
        if (priceContent) {
            priceContent.innerHTML = `<p>Comparing prices with local retailers...</p>`;
        }

        const stockContent = document.getElementById('stock-content');
        if (stockContent) {
            stockContent.innerHTML = `
                <div class="trending-item">
                    <div class="trending-item-header">
                        <strong>Potatoes</strong>
                    </div>
                    <div class="trending-item-details">
                        <span>Stock: 50</span>
                        <span>₹1250.00</span>
                    </div>
                </div>
            `;
        }

        const alertsContent = document.getElementById('alerts-content');
        if (alertsContent) {
            alertsContent.innerHTML = `
                <div class="alert-item stock-alert">
                    <i class="fas fa-exclamation-triangle"></i>
                    <div>
                        <strong>Salt</strong>
                        <p>Low stock (5 remaining)</p>
                    </div>
                </div>
            `;
        }

        const recsContent = document.getElementById('recs-content');
        if (recsContent) {
            recsContent.innerHTML = `
                <div class="recommendation">
                    <i class="fas fa-shopping-basket"></i>
                    <div>
                        <strong>Buy in bulk</strong>
                        <p>Potatoes have 10% discount this week</p>
                    </div>
                </div>
            `;
        }
    }

    // Download report handler
    document.getElementById('download-report')?.addEventListener('click', function(e) {
        e.preventDefault();
        alert('This would generate and download a PDF report in the full implementation');
    });
});

// document.addEventListener('DOMContentLoaded', function() {
//     // Flash message auto-hide
//     const flashes = document.querySelectorAll('.flash');
//     if (flashes.length) {
//         flashes.forEach(flash => {
//             setTimeout(() => {
//                 flash.style.opacity = '0';
//                 setTimeout(() => flash.remove(), 300);
//             }, 3000);
//         });
//     }

//     // Check if we have uploaded data (simulated with sessionStorage)
//     const hasData = sessionStorage.getItem('hasData');
    
//     if (hasData) {
//         enableDashboard();
//         loadSampleData(); // Replace with actual data loading in your implementation
//     }

//     // Simulate upload completion (in real app, this would be after successful upload)
//     document.querySelector('.cta-button')?.addEventListener('click', function(e) {
//         if (!hasData) {
//             e.preventDefault();
//             sessionStorage.setItem('hasData', 'true');
//             enableDashboard();
//             loadSampleData(); // Replace with actual data loading
//             // In real app, you would redirect to upload page instead
//         }
//     });

//     function enableDashboard() {
//         // Enable navbar items
//         document.querySelectorAll('.nav-disabled').forEach(item => {
//             item.classList.remove('nav-disabled');
//         });

//         // Show action buttons
//         document.getElementById('results-actions').style.display = 'flex';

//         // Make cards clickable
//         document.querySelectorAll('.dashboard-card').forEach(card => {
//             card.style.cursor = 'pointer';
//             card.addEventListener('click', function() {
//                 const cardId = this.id;
//                 // In real app, load appropriate content for each card
//                 console.log(`Clicked ${cardId}`);
//                 // Example: window.location.href = `/price_comparisons`;
//             });
//         });
//     }

//     function loadSampleData() {
//         // This is just for demonstration - replace with actual data loading
//         document.getElementById('price-content').innerHTML = `
//             <p>Comparing prices with local retailers...</p>
//         `;
        
//         document.getElementById('stock-content').innerHTML = `
//             <div class="trending-item">
//                 <div class="trending-item-header">
//                     <strong>Potatoes</strong>
//                 </div>
//                 <div class="trending-item-details">
//                     <span>Stock: 50</span>
//                     <span>¥1250.00</span>
//                 </div>
//             </div>
//         `;
        
//         document.getElementById('alerts-content').innerHTML = `
//             <div class="alert-item stock-alert">
//                 <i class="fas fa-exclamation-triangle"></i>
//                 <div>
//                     <strong>Salt</strong>
//                     <p>Low stock (5 remaining)</p>
//                 </div>
//             </div>
//         `;
        
//         document.getElementById('recs-content').innerHTML = `
//             <div class="recommendation">
//                 <i class="fas fa-shopping-basket"></i>
//                 <div>
//                     <strong>Buy in bulk</strong>
//                     <p>Potatoes have 10% discount this week</p>
//                 </div>
//             </div>
//         `;
//     }

//     // Download report handler
//     document.getElementById('download-report')?.addEventListener('click', function(e) {
//         e.preventDefault();
//         alert('This would generate and download a PDF report in the full implementation');
//     });
// });