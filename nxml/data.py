
from nwebclient import runner as r
from nwebclient import base as b
from nwebclient import util as u

import imaplib

class Imap(r.BaseJobExecutor):
    def __init__(self, host, user, password, type='imap', args: u.Args={}):
        super().__init__(type)
        self.host = host
        self.user = user
        self.password = password

    def get_all(self, limit=100):
        res = []
        m = imaplib.IMAP4(host=self.host)
        m.login(self.user, self.password)
        m.select()
        typ, data = m.search(None, 'ALL')
        i = 0
        for num in data[0].split():
            typ, data = m.fetch(num, '(RFC822)')
            #print('Message %s\n%s\n' % (num, data[0][1]))
            res.append(dict(title=type, data=data[0][1]))
            if limit is not None and i >= limit:
                break
        m.close()
        m.logout()
        return res

    def part_index(self, p: b.Page, params={}):
        for m in self.get_all():
            p.pre(m)
