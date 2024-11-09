# RAG-Abfrage-Schnittstelle

Hier kannst du eine PDF hochladen und eine Frage dazu stellen.

<form id="ragForm" enctype="multipart/form-data">
    <label for="pdf">PDF hochladen:</label><br>
    <input type="file" id="pdf" name="file" accept="application/pdf" required><br><br>
    <label for="query">Deine Frage:</label><br>
    <input type="text" id="query" name="query" required><br><br>
    <button type="submit">Absenden</button>
</form>

<div id="response">
    <!-- Die Antwort wird hier angezeigt -->
</div>

<div id="loading-spinner" style="display: none; font-size: 18px; margin-top: 20px;">
    ⏳ Verarbeitung läuft, bitte warten...
</div>

<script>
    document.getElementById('ragForm').onsubmit = async (event) => {
        event.preventDefault();

        let formData = new FormData();
        formData.append("file", document.getElementById('pdf').files[0]);
        formData.append("query", document.getElementById('query').value);

        const spinner = document.getElementById('loading-spinner');
        const responseDiv = document.getElementById('response');

        // Lade-Spinner anzeigen
        spinner.style.display = "block";
        responseDiv.innerText = "";

        try {
            const response = await fetch("http://127.0.0.1:5000/upload", {
                method: "POST",
                body: formData,
            });

            const result = await response.json();
            if (response.ok) {
                responseDiv.innerText = "Antwort: " + result.answer;
            } else {
                responseDiv.innerText = "Fehler: " + result.error;
            }
        } catch (error) {
            responseDiv.innerText = "Netzwerkfehler: " + error.message;
        } finally {
            // Lade-Spinner ausblenden
            spinner.style.display = "none";
        }
    };
</script>
