package components

import "testing"

func TestTruncateLines_Basic(t *testing.T) {
	got := TruncateLines("hello world", 50, 10)
	if got != "hello world" {
		t.Errorf("expected 'hello world', got %q", got)
	}
}

func TestTruncateLines_Wraps(t *testing.T) {
	got := TruncateLines("abcdefghij", 5, 10)
	want := "abcde\nfghij"
	if got != want {
		t.Errorf("expected %q, got %q", want, got)
	}
}

func TestTruncateLines_TruncatesToMaxLines(t *testing.T) {
	got := TruncateLines("a\nb\nc\nd\ne", 50, 3)
	want := "c\nd\ne"
	if got != want {
		t.Errorf("expected %q, got %q", want, got)
	}
}

func TestTruncateLines_ZeroWidth(t *testing.T) {
	got := TruncateLines("hello", 0, 5)
	if got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

func TestTruncateLines_ZeroMaxLines(t *testing.T) {
	got := TruncateLines("hello", 50, 0)
	if got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

func TestTruncateLines_EmptyInput(t *testing.T) {
	got := TruncateLines("", 10, 5)
	if got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

func TestTruncateLines_WrapAndTruncate(t *testing.T) {
	// 20 chars at width 5 = 4 lines, maxLines 2 = last 2
	got := TruncateLines("abcdefghijklmnopqrst", 5, 2)
	want := "klmno\npqrst"
	if got != want {
		t.Errorf("expected %q, got %q", want, got)
	}
}
