// Downloads the text in source.data['text'][0] as a file named `filename`
// (args: source, filename). An empty source downloads nothing; the server
// counts the saves in source.data['save'], so that saving the same text
// twice changes the data and downloads twice.
const texts = source.data['text']
if (texts.length > 0) {
    const blob = new Blob([texts[0]], { type: 'text/yaml;charset=utf-8;' })

    //addresses IE
    if (navigator.msSaveBlob) {
        navigator.msSaveBlob(blob, filename)
    } else {
        const link = document.createElement('a')
        link.href = URL.createObjectURL(blob)
        link.download = filename
        link.target = '_blank'
        link.style.visibility = 'hidden'
        link.dispatchEvent(new MouseEvent('click'))
    }
}
