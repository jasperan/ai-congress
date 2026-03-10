package app

import "testing"

func TestLerpColor_Start(t *testing.T) {
	got := lerpColor("#000000", "#ffffff", 0)
	if got != "#000000" {
		t.Errorf("expected #000000, got %s", got)
	}
}

func TestLerpColor_End(t *testing.T) {
	got := lerpColor("#000000", "#ffffff", 1)
	if got != "#ffffff" {
		t.Errorf("expected #ffffff, got %s", got)
	}
}

func TestLerpColor_Mid(t *testing.T) {
	got := lerpColor("#000000", "#ff0000", 0.5)
	// 0 + (255-0)*0.5 = 127
	if got != "#7f0000" {
		t.Errorf("expected #7f0000, got %s", got)
	}
}

func TestHexToRGB_Valid(t *testing.T) {
	r, g, b := hexToRGB("#ff8800")
	if r != 255 || g != 136 || b != 0 {
		t.Errorf("expected (255,136,0), got (%d,%d,%d)", r, g, b)
	}
}

func TestHexToRGB_Invalid(t *testing.T) {
	r, g, b := hexToRGB("notahex")
	if r != 0 || g != 0 || b != 0 {
		t.Errorf("expected (0,0,0) for invalid, got (%d,%d,%d)", r, g, b)
	}
}

func TestPadRight_Short(t *testing.T) {
	got := padRight("abc", 6)
	if got != "abc   " {
		t.Errorf("expected 'abc   ', got %q", got)
	}
}

func TestPadRight_Exact(t *testing.T) {
	got := padRight("abcdef", 6)
	if got != "abcdef" {
		t.Errorf("expected 'abcdef', got %q", got)
	}
}

func TestPadRight_Long(t *testing.T) {
	got := padRight("abcdefgh", 6)
	if got != "abcdefgh" {
		t.Errorf("expected 'abcdefgh', got %q", got)
	}
}
