const searchBtn = document.querySelector('#search-btn');
const searchBar = document.querySelector('#search-bar');
const results = document.querySelector('#results');

searchBtn.addEventListener('click', () => {
  results.innerHTML = 'Loading...';

  fetch('/search', {
    method: 'POST',
    body: JSON.stringify({ query: searchBar.value }),
    headers: { 'Content-Type': 'application/json' },
  })
    .then((res) => res.json())
    .then((data) => {
      results.innerHTML = '';

      data.forEach((result) => {
        const div = document.createElement('div');
        div.classList.add('result');

        const img = document.createElement('img');
        img.src = result.image;

        const h2 = document.createElement('h2');
        h2.textContent
        h2.textContent = result.name;

        div.appendChild(img);
        div.appendChild(h2);
        results.appendChild(div);
      });
    })
    .catch((err) => {
      results.innerHTML = 'An error occurred.';
      console.error(err);
    });
    });
