package components

import "testing"

func TestSparkline_EmptyTracker(t *testing.T) {
	tp := NewThroughputTracker()
	s := tp.Sparkline(10)
	if len([]rune(s)) != 10 {
		t.Errorf("expected 10 runes, got %d", len([]rune(s)))
	}
}

func TestSparkline_ZeroWidth(t *testing.T) {
	tp := NewThroughputTracker()
	s := tp.Sparkline(0)
	if s != "" {
		t.Errorf("expected empty, got %q", s)
	}
}

func TestSparkline_NegativeWidth(t *testing.T) {
	tp := NewThroughputTracker()
	s := tp.Sparkline(-5)
	if s != "" {
		t.Errorf("expected empty, got %q", s)
	}
}

func TestCurrentRate_EmptyTracker(t *testing.T) {
	tp := NewThroughputTracker()
	rate := tp.CurrentRate()
	if rate != 0 {
		t.Errorf("expected 0, got %f", rate)
	}
}

func TestAdd_IncrementsRate(t *testing.T) {
	tp := NewThroughputTracker()
	tp.Add(10)
	rate := tp.CurrentRate()
	if rate < 1 {
		t.Errorf("expected rate > 0 after Add, got %f", rate)
	}
}
