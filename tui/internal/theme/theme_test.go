package theme

import "testing"

func TestConfidenceColor_High(t *testing.T) {
	c := ConfidenceColor(0.9)
	if c != Accent {
		t.Errorf("expected Accent for 0.9, got %v", c)
	}
}

func TestConfidenceColor_Medium(t *testing.T) {
	c := ConfidenceColor(0.5)
	if c != Warning {
		t.Errorf("expected Warning for 0.5, got %v", c)
	}
}

func TestConfidenceColor_Low(t *testing.T) {
	c := ConfidenceColor(0.2)
	if c != Danger {
		t.Errorf("expected Danger for 0.2, got %v", c)
	}
}

func TestConfidenceColor_Boundary(t *testing.T) {
	if ConfidenceColor(0.7) != Accent {
		t.Error("0.7 should be Accent (high)")
	}
	if ConfidenceColor(0.4) != Warning {
		t.Error("0.4 should be Warning (medium)")
	}
	if ConfidenceColor(0.39) != Danger {
		t.Error("0.39 should be Danger (low)")
	}
}

func TestStatusColor_Streaming(t *testing.T) {
	if StatusColor("streaming") != Accent {
		t.Error("streaming should return Accent")
	}
}

func TestStatusColor_Error(t *testing.T) {
	if StatusColor("error") != Danger {
		t.Error("error should return Danger")
	}
}

func TestStatusColor_Idle(t *testing.T) {
	if StatusColor("idle") != Muted {
		t.Error("idle should return Muted")
	}
}

func TestStatusColor_Unknown(t *testing.T) {
	if StatusColor("something_else") != Muted {
		t.Error("unknown status should return Muted")
	}
}
