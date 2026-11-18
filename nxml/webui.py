#
# Project: nxml
#
# Gemeinsame Bausteine für Runner-Seiten
#
import json

RUNNER_NAV_JS = r"""
(function() {
  // UI v2 (/pysys/ui): Seiten des Runners im Inhaltsbereich laden, sonst normale Links
  const T = RUNNER_NAV_TYPE;
  const navs = document.querySelectorAll('nav.runner-nav[data-type="' + T + '"]');
  const nav = navs[navs.length - 1], content = document.getElementById('content');
  if (!nav || !content || !content.contains(nav) || location.pathname.indexOf('/pysys/ui') !== 0) return;
  nav.addEventListener('click', (e) => {
    const a = e.target.closest('a');
    if (!a) return;
    const page = new URL(a.href, location.href).searchParams.get(T);
    if (!page) return;
    e.preventDefault();
    fetch('/', { method: 'POST', body: new URLSearchParams({ type: T, page }) }).then((r) => r.json()).then((r) => {
      if (!r.success) { a.title = r.message || 'Fehler'; return; }
      content.innerHTML = r.content + (r.intern_links || '');
      content.querySelectorAll('script').forEach((old) => {
        const s = document.createElement('script');
        [...old.attributes].forEach((at) => s.setAttribute(at.name, at.value));
        s.textContent = old.textContent;
        old.replaceWith(s);
      });
    });
  });
})();
"""


def part_nav(runner, p, pages, active='index', css_class=''):
    """
    Navigation zwischen den Seiten eines Runners (p.nav). Im klassischen UI normale Links auf
    /pysys/dispatcher?type=<type>&<type>=<seite>, in der UI v2 werden die Seiten im Inhaltsbereich gewechselt.

    :param pages: [(seite, titel)], seite = Name von part_<seite> (index = Startseite)
    """
    p.nav(*[(title, runner.dispatcher_link(name)) for name, title in pages], active=runner.dispatcher_link(active),
          _class=('runner-nav ' + css_class).strip(), data_type=runner.type)
    p('<script>{ const RUNNER_NAV_TYPE = ' + json.dumps(runner.type) + ';\n' + RUNNER_NAV_JS + '\n}</script>')
