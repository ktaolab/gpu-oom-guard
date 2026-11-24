import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gpu_oom_guard as g

GiB = g.GiB

def fake_reader(table):
    return lambda pid: (g.Proc(pid, *table[pid], "cmd") if pid in table else None)

class TierTests(unittest.TestCase):
    cfg = g.Config()
    def t(self, cg): return g.tier_of(cg, self.cfg)
    def test_tmux_is_adhoc(self):
        self.assertEqual(self.t("/user.slice/user-1000.slice/user@1000.service/tmux-spawn-abc.scope"), 0)
    def test_ssh_session_is_adhoc(self):
        self.assertEqual(self.t("/user.slice/user-1000.slice/session-4.scope"), 0)
    def test_docker_before_scope(self):
        self.assertEqual(self.t("/system.slice/docker-0123abcd.scope"), 1)
    def test_ollama(self):
        self.assertEqual(self.t("/system.slice/ollama.service"), 1)
    def test_user_service(self):
        self.assertEqual(self.t("/user.slice/user-1000.slice/user@1000.service/app.slice/comfyui.service"), 2)
    def test_unmatched_default(self):
        self.assertEqual(self.t("/"), 0)

class RankTests(unittest.TestCase):
    def test_adhoc_before_bigger_service(self):
        table = {10: ("python", "/system.slice/big.service"),
                 11: ("python", "/user.slice/user-1000.slice/session-1.scope"),
                 12: ("llama-server", "/system.slice/ollama.service")}
        usage = {10: 47 * GiB, 11: 2 * GiB, 12: 30 * GiB}
        r = g.rank_victims(usage, g.Config(), reader=fake_reader(table))
        self.assertEqual([p.pid for _, _, p in r], [11, 12, 10])
    def test_largest_within_tier(self):
        table = {1: ("a", "/x/a.scope"), 2: ("b", "/x/b.scope")}
        r = g.rank_victims({1: GiB, 2: 5 * GiB}, g.Config(), reader=fake_reader(table))
        self.assertEqual(r[0][2].pid, 2)
    def test_protected_and_gone_and_init(self):
        table = {5: ("Xorg", "/x/a.scope"), 6: ("job", "/x/b.scope")}
        r = g.rank_victims({1: 9 * GiB, 5: 9 * GiB, 6: GiB, 7: 9 * GiB}, g.Config(), reader=fake_reader(table))
        self.assertEqual([p.pid for _, _, p in r], [6])

class DangerTests(unittest.TestCase):
    cfg = g.Config()
    def test_plenty(self): self.assertFalse(g.in_danger(50 * GiB, 90, self.cfg))
    def test_low_but_calm(self): self.assertFalse(g.in_danger(5 * GiB, 1, self.cfg))
    def test_low_and_pressured(self): self.assertTrue(g.in_danger(5 * GiB, 30, self.cfg))
    def test_hard_floor(self): self.assertTrue(g.in_danger(2 * GiB, 0, self.cfg))

class ParseTests(unittest.TestCase):
    def test_nvidia_smi(self):
        self.assertEqual(g.parse_nvidia_smi("2421, 48225\n2431, 7702\n99, [N/A]\n"),
                         {2421: 48225 * 1024 * 1024, 2431: 7702 * 1024 * 1024})
    def test_psi(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("some avg10=1.00 avg60=0.00 avg300=0.00 total=1\nfull avg10=42.50 avg60=0 avg300=0 total=1\n")
        self.assertEqual(g.read_psi_full_avg10(f.name), 42.5); os.unlink(f.name)
    def test_psi_missing(self):
        self.assertEqual(g.read_psi_full_avg10("/nonexistent"), 0.0)

@unittest.skipIf(sys.version_info < (3, 11), "tomllib")
class ConfigTests(unittest.TestCase):
    def test_toml(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write('kill_avail_gib = 8\nprotected_extra = ["myd"]\n'
                    '[[tier_rules]]\nmatch = "*/critical.service"\ntier = 9\n')
        cfg = g.load_toml(f.name, g.Config()); os.unlink(f.name)
        self.assertEqual(cfg.kill_avail_gib, 8)
        self.assertIn("myd", cfg.protected); self.assertIn("Xorg", cfg.protected)
        self.assertEqual(g.tier_of("/system.slice/critical.service", cfg), 9)

if __name__ == "__main__":
    unittest.main()
