param(
    [Parameter(Mandatory = $true)][string]$Docx,
    [Parameter(Mandatory = $true)][string]$KeysFile,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Pdf = ""
)
# Ask Word for the displayed page number of every paragraph whose text equals one of the keys (outside tables),
# optionally export a PDF. Used by make_srm_report.py to fill the contents and the lists of figures and tables.
Add-Type -AssemblyName System.Web.Extensions
$list = (New-Object System.Web.Script.Serialization.JavaScriptSerializer).Deserialize((Get-Content -Raw -Encoding UTF8 $KeysFile), [object])
$res = [ordered]@{}
$trim = [char[]]@(13, 7, 32, 9, 160)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
$d = $null
try {
    $d = $w.Documents.Open($Docx, $false, $true)
    $d.Repaginate()
    $end = $d.Content.End
    foreach ($key in $list) {
        $needle = [string]$key
        if ($needle.Length -gt 240) { $needle = $needle.Substring(0, 240) }
        else { $needle = $needle + "^p" }  # ^p: whole paragraph only, so table-cell copies are not matched
        $rng = $d.Range(0, $end)
        $f = $rng.Find
        $f.ClearFormatting()
        $f.Text = $needle
        $f.Forward = $true
        $f.Wrap = 0
        $f.MatchCase = $true
        $f.MatchWildcards = $false
        $guard = 0
        while ($f.Execute() -and $guard -lt 20) {
            $guard++
            $para = $rng.Paragraphs.Item(1).Range.Text.Trim($trim)
            if (-not $rng.Information(12) -and ($para -eq [string]$key -or ($para.StartsWith([string]$key) -and $para.Length -le ([string]$key).Length + 4))) {
                $res[[string]$key] = [int]$rng.Information(1)
                break
            }
            $rng.SetRange($rng.End, $end)
            $f = $rng.Find
            $f.ClearFormatting(); $f.Text = $needle; $f.Forward = $true; $f.Wrap = 0; $f.MatchCase = $true; $f.MatchWildcards = $false
        }
    }
    $res | ConvertTo-Json | Set-Content -Encoding UTF8 $Out
    "pages found: " + $res.Count + " of " + $list.Count + "; total pages " + $d.ComputeStatistics(2)
    if ($Pdf) { $d.ExportAsFixedFormat($Pdf, 17) }
} finally {
    if ($d) { try { $d.Close($false) } catch {} }
    try { $w.Quit() } catch {}
}
