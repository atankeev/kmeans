package coverageprobe

import "testing"

func TestClamp(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		give int
		want int
	}{
		{name: "within", give: 5, want: 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := Clamp(tt.give); got != tt.want {
				t.Fatalf("Clamp(%d) = %d, want %d", tt.give, got, tt.want)
			}
		})
	}
}
