// Downloads the columns `columns` of `source`, in this order, as a CSV file
// named `filename` (args: source, filename, columns). The order must not
// come from the source: under Bokeh 3 the browser's order of its columns
// follows the server's string hash seed. If a column is missing, the file
// is not downloaded and the error is logged in the browser console (the
// other downloads of the button still run).
function table_to_csv(source, columns) {
    const missing = columns.filter(column => !(column in source.data))
    if (missing.length > 0) {
        const message = filename + ': the source has no column ' +
            missing.join(', ') + ' (it has ' +
            Object.keys(source.data).join(', ') + ')'
        console.error(message)
        return null
    }
    const nrows = source.get_length()
    const lines = [columns.join(',')]

    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < columns.length; j++) {
            const column = columns[j]
            row.push(source.data[column][i].toString())
        }
        lines.push(row.join(','))
    }
    return lines.join('\n').concat('\n')
}

const filetext = table_to_csv(source, columns)
if (filetext !== null) {
    const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' })

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
