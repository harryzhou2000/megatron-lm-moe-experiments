partitions_B300 = [
    "b300@ts6/dgx-b300@ts1/8gpu-256cpu-2048gb",
    "b300@ts6/dgx-b300@ts1/8gpu-256cpu-2048gb",
    "b300@ts7/dgx-b300@ts1/8gpu-256cpu-2048gb",
    "b300@ts7/dgx-b300@ts1/8gpu-256cpu-2048gb",
    "b300@ts7/dgx-b300@ts1/8gpu-256cpu-2048gb",
    "b300@ts7/dgx-b300@ts1/8gpu-256cpu-2048gb",
    "b300@ts7/dgx-b300@ts1/8gpu-256cpu-2048gb",
    "b300@ts4/b300-nvl8@ts3/8gpu-224cpu-2048gb",
    "b300@ts8/b300-nvl8@cr+mp/8gpu-224cpu-2048gb",
    "b300@ts4/b300-nvl8@ts3/8gpu-224cpu-2048gb",
    "b300@ts4/b300-nvl8@ts3/8gpu-224cpu-2048gb",
    "b300@ts4/b300-nvl8@ts3/8gpu-224cpu-2048gb",
    "b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb",
    "b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb",
    "b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb",
    "b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb",
    "b300@qs1/b300-nvl8@cr+mp/8gpu-224cpu-2048gb",
    "b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb",
    "b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb",
    "b300@ts7/b300-nvl8@ts7/8gpu-224cpu-2048gb",
    "b300@ts7/b300-nvl8@ts7/8gpu-224cpu-2048gb",
]

partitions_B200 = [
    "b200@ts7-e00-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@ts-a01p-1000W/umbriel-b200@ts5/8gpu-224cpu-2048gb",
    "ovsb/b200@ts-a01p-1000W/umbriel-b200@ts5/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "ovsb/b200@ts-a01p-1000W/umbriel-b200@ts5/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
    "b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
]

partitions_B200 += [
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@501-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
      'b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb',
]

partitions_B300 = list(set(partitions_B300))
partitions_B200 = list(set(partitions_B200))

print(partitions_B300)

print(partitions_B200)