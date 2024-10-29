# RAG Query Interface

<form id="ragForm" enctype="multipart/form-data">
    <label for="pdf">Upload PDF:</label>
    <input type="file" id="pdf" name="file" accept="application/pdf" required>
    <label for="query">Enter your query:</label>
    <input type="text" id="query" name="query" required>
    <button type="submit">Submit</button>
</form>

<div id="response">
    <!-- Display response here -->
</div>

<script>
    document.getElementById('ragForm').onsubmit = async (event) => {
        event.preventDefault();

        let formData = new FormData();
        formData.append("file", document.getElementById('pdf').files[0]);
        formData.append("query", document.getElementById('query').value);

        const response = await fetch("http://127.0.0.1:5000/upload", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();
        document.getElementById('response').innerText = "Answer: " + result.answer;
    };
</script>
