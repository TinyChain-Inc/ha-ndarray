use std::fmt;

use memoize::memoize;

use super::{build, ArrayFormat, Program};
use crate::ops::SliceSpec;
use crate::{AxisRange, Error};

#[derive(Clone, Eq, PartialEq, Hash)]
struct RangeFormat<'a> {
    axes: &'a [AxisRange],
}

impl<'a> RangeFormat<'a> {
    fn max_indices(&self) -> usize {
        self.axes
            .iter()
            .map(|bound| match bound {
                AxisRange::At(_) => 1,
                AxisRange::In(_, _, _) => 0,
                AxisRange::Of(indices) => indices.len(),
            })
            .fold(1, Ord::max)
    }
}

impl<'a> fmt::Display for RangeFormat<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("{ ")?;
        if self.axes.is_empty() {
            f.write_str("0")?;
        }

        for bound in self.axes {
            f.write_str("{ ")?;

            match bound {
                AxisRange::At(i) => {
                    write!(f, ".btype=AT, ")?;
                    write!(f, "{{ .at_index={i} }}")?;
                }
                AxisRange::In(start, stop, step) => {
                    write!(f, ".btype=IN, ")?;
                    write!(f, "{{ .in_range={{ {start}, {stop}, {step}, 0 }} }}")?;
                }
                AxisRange::Of(indices) => {
                    write!(f, ".btype=OF, ")?;

                    f.write_str("{ .of_indices={ ")?;

                    for i in indices {
                        write!(f, "{i}, ")?;
                    }

                    f.write_str("} }")?;
                }
            }

            f.write_str(" }, ")?;
        }

        f.write_str(" }")
    }
}

// TODO: use the SharedCache option
#[memoize(Capacity: 1024)]
pub fn read_slice(c_type: &'static str, spec: SliceSpec) -> Result<Program, Error> {
    let ndim = spec.shape().len();
    let source_ndim = spec.range().len();

    let dims = ArrayFormat::from(spec.shape().as_slice());
    let strides = ArrayFormat::from(spec.strides().as_slice());

    let bounds = RangeFormat {
        axes: spec.range().as_slice(),
    };

    let source_strides = ArrayFormat::from(spec.source_strides().as_slice());

    let src = format!(
        r#"
        typedef enum {{ AT, IN, OF }} RangeType;

        typedef union {{
            ulong at_index;
            ulong4 in_range;
            ulong of_indices[{max_indices}];
        }} AxisRange;

        typedef struct {{
            RangeType btype;
            AxisRange range;
        }} Bound;

        const ulong dims[{ndim_slots}] = {dims};
        const ulong strides[{ndim_slots}] = {strides};

        const Bound bounds[{source_slots}] = {bounds};
        const ulong source_strides[{source_slots}] = {source_strides};

        __kernel void read_slice(
                __global const {c_type}* restrict input,
                __global {c_type}* restrict output)
        {{
            const ulong offset_out = get_global_id(0);

            ulong coord[{ndim_slots}];

            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                coord[x] = strides[x] == 0 ? 0 : (offset_out / strides[x]) % dims[x];
            }}

            ulong offset_in = 0;
            ulong x = 0;

            #pragma unroll
            for (uint source_x = 0; source_x < {source_ndim}; source_x++) {{
                ulong i = 0;

                switch (bounds[source_x].btype) {{
                    case AT:
                        i = bounds[source_x].range.at_index;

                        break;
                    case IN:
                        i = (
                            bounds[source_x].range.in_range.x +
                            (coord[x] * bounds[source_x].range.in_range.z)
                        );

                        x++;
                        break;
                    case OF:
                        i = bounds[source_x].range.of_indices[coord[x]];

                        x++;
                        break;
                }}

                offset_in += source_strides[source_x] * i;
            }}

            output[offset_out] = input[offset_in];
        }}
        "#,
        max_indices = bounds.max_indices(),
        ndim_slots = ndim.max(1),
        source_slots = source_ndim.max(1),
    );

    build(&src, &[c_type], "slice")
}
