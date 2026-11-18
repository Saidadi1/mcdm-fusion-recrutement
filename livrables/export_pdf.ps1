# Met à jour la table des matières du rapport, exporte rapport et présentation en PDF (Office requis).
#   powershell -ExecutionPolicy Bypass -File livrables\export_pdf.ps1
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$docx = Join-Path $dir "Rapport_Fusion_Avis_MCDM.docx"
$pptx = Join-Path $dir "Presentation_Fusion_Avis_MCDM.pptx"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open($docx)
$doc.TablesOfContents | ForEach-Object { $_.Update() }
$doc.Fields.Update() | Out-Null
$doc.Save()
$doc.SaveAs([ref](Join-Path $dir "Rapport_Fusion_Avis_MCDM.pdf"), [ref]17)
"Pages du rapport : " + $doc.ComputeStatistics(2)
$doc.Close()
$word.Quit()

try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $pres = $ppt.Presentations.Open($pptx, $true, $false, $false)
    $pres.SaveAs((Join-Path $dir "Presentation_Fusion_Avis_MCDM.pdf"), 32)
    "Diapositives : " + $pres.Slides.Count
    $pres.Close()
    $ppt.Quit()
} catch { "PowerPoint indisponible : export PDF de la présentation ignoré" }
